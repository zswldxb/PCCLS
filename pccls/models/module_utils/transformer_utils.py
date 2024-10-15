import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from pccls.models.module_utils.act_utils import act_fn


class MultiHeadAttn(nn.Module):
    def __init__(self, d_f, n_h, dropout=0., bias=True):
        super().__init__()
        assert d_f % n_h == 0
        self.d_f = d_f
        self.scale = d_f ** -0.5
        self.n_h = n_h
        self.d_h = d_f // n_h

        self.w_q = nn.Linear(d_f, d_f, bias=bias)
        self.w_k = nn.Linear(d_f, d_f, bias=bias)
        self.w_v = nn.Linear(d_f, d_f, bias=bias)
        self.w_o = nn.Linear(d_f, d_f, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q, k, v, *, key_mask=None, attn_mask=None):
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3 and k.shape == v.shape

        b_s, n_q, n_k = q.size(0), q.size(1), k.size(1)
        d_f, n_h, d_h = self.d_f, self.n_h, self.d_h

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3) and attn_mask.dtype == torch.bool
            if attn_mask.dim() == 2:
                assert attn_mask.shape == (n_q, n_k)
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                assert attn_mask.shape == (b_s * n_h, n_q, n_k)

        if key_mask is not None:
            assert key_mask.dim() == 2 and key_mask.dtype == torch.bool and key_mask.shape == (b_s, n_k)
            key_mask = key_mask.view(b_s, 1, 1, n_k).expand(-1, n_h, -1, -1).reshape(b_s * n_h, 1, n_k)
            if attn_mask is None:
                attn_mask = key_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_mask)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(attn_mask, float("-inf"))

        q = rearrange(self.w_q(q), "b n (h d) -> (b h) n d", b=b_s, n=n_q, h=n_h, d=d_h).contiguous()
        k_t = rearrange(self.w_k(k), "b n (h d) -> (b h) d n", b=b_s, n=n_k, h=n_h, d=d_h).contiguous()
        v = rearrange(self.w_v(v), "b n (h d) -> (b h) n d", b=b_s, n=n_k, h=n_h, d=d_h).contiguous()

        attn = (self.scale * q) @ k_t
        if attn_mask is not None:
            attn = attn + attn_mask
        x = self.dropout(self.softmax(attn)) @ v
        x = rearrange(x, '(b h) n d -> b n (h d)', b=b_s, n=n_q, h=n_h, d=d_h).contiguous()
        x = self.w_o(x)
        return x


class GroupAttention(nn.Module):
    def __init__(self, d_feat, d_ffn, n_head, drop_attn=0.1, drop_ffn=0.0, act='relu') -> None:
        super().__init__()
        self.attn = MultiHeadAttn(d_feat, n_head, drop_attn=drop_attn)
        self.norm1 = nn.LayerNorm(d_feat)
        self.ffn = nn.Sequential(nn.Linear(d_feat, d_ffn), nn.Dropout(drop_ffn), act_fn(act), nn.Linear(d_ffn, d_feat))
        self.norm2 = nn.LayerNorm(d_feat)

    def forward(self, feat, p2g_idx, g2p_idx, pig_msk, pos_emb):
        residual = feat
        grp_fea, grp_pos = feat[p2g_idx], pos_emb[p2g_idx]
        q = k = grp_fea + grp_pos
        v = grp_fea
        feat = self.attn(q, k, v, key_mask=pig_msk)[g2p_idx]
        feat = self.norm1(feat + residual)

        residual = feat
        feat = self.ffn(feat)
        feat = self.norm2(feat + residual)

        return feat


class GroupEncoderLayer(nn.Module):
    def __init__(self, d_feat, d_ffn, n_head, drop_attn=0.1, drop_ffn=0.0, act="relu"):
        super().__init__()
        self.attn = GroupAttention(d_feat, d_ffn, n_head, drop_attn, drop_ffn, act)
        self.norm = nn.LayerNorm(d_feat)

    def forward(self, feat, v2g_idx, g2v_idx, vig_msk, pos_emb):
        residual = feat
        feat = self.attn(feat, v2g_idx, g2v_idx, vig_msk, pos_emb)
        feat = self.norm(feat + residual)
        return feat