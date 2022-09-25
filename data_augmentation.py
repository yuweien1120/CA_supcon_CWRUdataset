import random

import torch
from torchvision.transforms import transforms

class Add_Gaussian:
    """添加高斯噪声"""
    def __init__(self, p: float=0.5, mean=0, std=0.05):
        """
        初始化高斯噪声相关参数
        :param p: 添加高斯噪声的概率
        :param mean: 均值
        :param std: 标准差
        """
        self.p = p
        # 先转为tensor类型，因为输入数据是张量，跟他进行运算也要是张量类型
        # as_tensor是浅拷贝
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)

        # 将输入的均值和标准差转化为列向量，后面可以利用广播机制
        self.mean = _covert_tensor_shape(mean)
        self.std = _covert_tensor_shape(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1)) < self.p:
            row, col = x.shape
            noise = torch.randn(row, col) * self.std + self.mean
            x = x + noise
        return x

class Random_Scale:
    """随机信号缩放"""
    def __init__(self, p: float=0.5, std: float=0.05):
        self.p = p
        std = torch.as_tensor(std)
        self.std = _covert_tensor_shape(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1)) < self.p:
            row, col = x.shape
            scale = torch.randn(row, col) * self.std + 1
            x = x * scale
        return x

class Mask_Noise:
    """添加掩码噪声（掩码噪声就是随机让一些元素置零）"""
    def __init__(self, p: float=0.5, u: float=0.1):
        self.p = p
        self.u = u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1)) < self.p:
            row, col = x.shape
            Mask = torch.rand(row, col)
            Mask = Mask.ge(self.u).float()
            x = x * Mask
        return x

class Translation:
    """随机将信号某一位置两边的序列调换位置"""
    def __init__(self, p: float=0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1)) < self.p:
            row, col = x.shape
            i = random.randint(0, col-1)
            x = torch.roll(x, shifts=i, dims=1)
        return x


def _covert_tensor_shape(x: torch.Tensor) -> torch.Tensor:
    """
    如果输入是一维张量，则将其转置为列向量（二维张量）
    :param x: 输入张量
    :return: 列向量
    """
    if x.ndim == 1:
        x = x.reshape((len(x),1))
    return x

if __name__ == "__main__":
    a = torch.rand(4,4)
    b = torch.rand(4,4)
    c = torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
    d = torch.unbind(c, dim=0)
    e = torch.max(a, dim=1, keepdim=True)
    print(c)
    print(d)

