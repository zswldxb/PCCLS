
import torch
import torch.nn as nn
import torch.nn.functional as F
from module_utils import PointClsHead


class PointNet(nn.Module):
    def __init__(self, cls_num=40):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        # self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(1024)
        self.head = PointClsHead(1024, cls_num)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.feat(x)
        x = self.head(x)
        return x



if __name__ == '__main__':
    import torch
    batch_size, channel_size, class_size, point_size = (1 << 4), 3, 40, (1 << 10)
    x, y = torch.rand((batch_size, channel_size, point_size)), torch.randint(0, 40, (batch_size,)).long()
    model = PointNet(40, False)
    loss = get_loss()
    z = model(x)
    print(y.shape, z.shape)
    loss(z, y)