import torch
import pccls.ops.bpa_point_aggregation_ops.bpa_point_aggregation_op as bpa
import pccls.ops.bpa_point_aggregation_ops.bpa_point_aggregation_utils as spa
from typing import List, Tuple, Dict, Callable
Tensor = torch.Tensor
import time
'''
furthest_point_sampling is ok
knn_query is ok
ball_query is ok
gather_query is ok
gather_group is ok
three_nn_query is ok
'''


def gather_points(pts: Tensor, idx: Tensor) -> Tensor:
    '''
    Description:
        Gather points from index.
    Input:
        pts: the values (features or coordinates) of points. Shape: (batch_size, num_pts, dim_pts).
        idx: the indices of gathered points. Shape: (batch_size, num_qry, [num_grp]).
    Output:
        res: the values (features or coordinates) of gathered points. Shape: (batch_size, num_qry, [num_grp], dim_pts).
    '''
    sz = idx.size()
    res = torch.gather(pts, 1, idx.reshape(sz[0], -1, 1).expand(-1, -1, pts.size(-1))).reshape(*sz, -1)
    return res


# ---------------------------------------- Sorting Coordinates Expansion Operations ---------------------------------------- #
def torch_lexsort(x: Tensor) -> Tensor:
    # Reference from https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850
    # x: (b_s, n_p, d_c)
    assert x.ndim == 3
    idx = torch.stack([torch.argsort(torch.unique(e, dim=-1, sorted=True, return_inverse=True)[1]) for e in x.permute(0, 2, 1)])
    return idx


def SortCoord3DGetIdx(self, z: Tensor, order='xyz') -> Tuple[Tensor, Tensor]:                                      # (b_s, n_p, d_c)
        b_s, n_p, _ = z.size()
        if order == 'yxz':
            z = z[:, :, [1, 0, 2]]
        idx_pa = torch_lexsort(z)                                                               # (b_s, n_p)
        idx_rg = torch.arange(n_p, device=z.device).reshape(1, n_p, 1).expand(b_s, -1, -1)      # (b_s, n_p, 1)
        idx_re = torch.sort(bpa.gather_query(idx_rg, idx_pa).reshape(b_s, n_p), dim=1)[1]            # (b_s, n_p)
        return idx_pa, idx_re                                                                   # (b_s, n_p), (b_s, n_p)


if __name__ == '__main__':
    b_s, c_s, p_n, g_n, g_k, g_r = 4, 3, (1<<13), (1<<10), (1<<5), 0.2
    bpa_pnt = torch.rand((b_s, p_n, c_s)).cuda()
    spa_pnt = bpa_pnt.clone().reshape(b_s*p_n, c_s)
    p_n_cnt = torch.tensor([p_n for i in range(b_s)], dtype=torch.int, device='cuda')
    g_n_cnt = torch.tensor([g_n for i in range(b_s)], dtype=torch.int, device='cuda')
    p_n_tot = torch.sum(p_n_cnt).item()
    g_n_tot = torch.sum(g_n_cnt).item()

    idx_qry = bpa.bpa_furthest_point_sampling(bpa_pnt, g_n)
    bpa_qry = bpa.bpa_gather_query(bpa_pnt, idx_qry)

    bpa_grp_idx_knn_1 = bpa.bpa_knn_query(bpa_pnt, bpa_qry, g_k)
    bpa_grp_idx_knn_2 = spa.knn_query(bpa_pnt, bpa_qry, g_k)
    print((bpa_grp_idx_knn_1 == bpa_grp_idx_knn_2).all())
    bpa_grp_knn_1 = bpa.bpa_gather_group(bpa_pnt, bpa_grp_idx_knn_1)
    bpa_grp_knn_2 = bpa.bpa_gather_group(bpa_pnt, bpa_grp_idx_knn_2)
    x1 = spa.square_distance(bpa_grp_knn_1.view(b_s*g_n, -1, c_s), bpa_qry.view(b_s*g_n, -1, c_s))
    x2 = spa.square_distance(bpa_grp_knn_2.view(b_s*g_n, -1, c_s), bpa_qry.view(b_s*g_n, -1, c_s))
    print(x1.shape, x2.shape)
    print((x1 - x2).abs().sum())

    bpa_grp_idx_ball_1 = bpa.bpa_ball_query(bpa_pnt, bpa_qry, g_k, g_r)
    bpa_grp_idx_ball_2 = spa.ball_query(bpa_pnt, bpa_qry, g_k, g_r)
    print((bpa_grp_idx_ball_1 == bpa_grp_idx_ball_2).all())
    bpa_grp_ball_1 = bpa.bpa_gather_group(bpa_pnt, bpa_grp_idx_ball_1)
    bpa_grp_ball_2 = bpa.bpa_gather_group(bpa_pnt, bpa_grp_idx_ball_2)
    x1 = spa.square_distance(bpa_grp_ball_1.view(b_s*g_n, -1, c_s), bpa_qry.view(b_s*g_n, -1, c_s))
    x2 = spa.square_distance(bpa_grp_ball_2.view(b_s*g_n, -1, c_s), bpa_qry.view(b_s*g_n, -1, c_s))
    print(x1.shape, x2.shape)
    print((x1 - x2).abs().sum())

    # bpa_grp_idx_ball = bpa.bpa_ball_query(bpa_pnt, bpa_qry, g_k, g_r)
    # spa_grp_idx_ball = bpa.spa_ball_query(spa_pnt, p_n_cnt, spa_qry, g_n_cnt, g_k, g_r)
    # print((bpa_grp_idx_ball.flatten() == (spa_grp_idx_ball.reshape(b_s, -1) - torch.arange(b_s, dtype=torch.int, device='cuda').reshape(b_s, -1).expand(-1, g_n*g_k) * p_n).flatten()).all())

    # bpa_grp_ball = bpa.bpa_gather_group(bpa_pnt, bpa_grp_idx_ball)
    # spa_grp_ball = bpa.spa_gather_group(spa_pnt, p_n_cnt, spa_grp_idx_ball, g_n_cnt)
    # print((bpa_grp_ball.flatten() == spa_grp_ball.flatten()).all())


    bpa_dis, bpa_idx = bpa.bpa_three_query(bpa_qry, bpa_pnt)
    # spa_dis, spa_idx = bpa.spa_three_nn_query(spa_qry, g_n_cnt, spa_pnt, p_n_cnt)
    # print((bpa_dis.flatten() == spa_dis.flatten()).all())
    # print((bpa_idx.flatten() == (spa_idx.reshape(b_s, -1) - torch.arange(b_s, dtype=torch.int, device='cuda').reshape(b_s, -1).expand(-1, g_n*3) * p_n).flatten()).all())

    dist_recip = 1.0 / (bpa_dis + 1e-8)
    bpa_bse = dist_recip / torch.sum(dist_recip, dim=-1, keepdim=True)
    print(bpa_dis.size(), bpa_idx.size(), bpa_bse.size())

    # dist_recip = 1.0 / (spa_dis + 1e-8)
    # spa_bse = dist_recip / torch.sum(dist_recip, dim=-1, keepdim=True)
    # print(spa_dis.size(), spa_idx.size(), spa_bse.size())

    # bpa_fea = bpa.bpa_three_interpolate(bpa_pnt, bpa_idx, bpa_bse)
    # spa_fea = bpa.spa_three_interpolate(spa_pnt, spa_idx, spa_bse)
    # print((bpa_fea.flatten() == spa_fea.flatten()).all())
