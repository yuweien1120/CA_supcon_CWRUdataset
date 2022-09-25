import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, kernel_size):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        length = x.size()[-1]

        out = F.relu(self.bn1(self.conv1(x)))
        out = out[:, :, :length]
        out = self.bn2(self.conv2(out))
        out = out[:, :, :length]

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetCWRU(nn.Module):
    def __init__(self, block, seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane):
        super(ResNetCWRU, self).__init__()
        self.seq_len = seq_len
        self.planes = planes
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(1, self.planes[0], kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(self.planes[0])
        self.layer = self._make_layer(block, num_blocks, pool_size)

        self.linear = nn.Linear(self.planes[-1] * (self.seq_len // (pool_size ** (num_blocks - 1))), linear_plane)

    def _make_layer(self, block, num_blocks, pool_size):
        layers = []

        for i in range(num_blocks - 1):
            layers.append(block(self.planes[i], self.planes[i + 1], kernel_size=self.kernel_size))
            layers.append(nn.MaxPool1d(pool_size))

        layers.append(block(self.planes[-2], self.planes[-1], kernel_size=self.kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = out[:, :, :self.seq_len]
        out = self.layer(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear(out))

        return out


def create_ResNetCWRU(seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane):
    net = ResNetCWRU(BasicBlock, seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane=linear_plane)
    return net

if __name__ == '__main__':
    from torchsummary import summary
    ResNetCWRU_params = {'seq_len': 400, 'num_blocks': 2, 'planes': [10, 10, 10],
                        'kernel_size': 10, 'pool_size': 2, 'linear_plane': 100}
    net = create_ResNetCWRU(**ResNetCWRU_params)

    summary(net, (1, 400))