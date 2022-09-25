import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        计算有监督对比损失
        :param features: 输入样本对后网络生成的所有归一化特征（shape:[2*batch_size, featyre_num]）
        :param labels: 特征标签（shanpe:[2*batch_size, 1]）
        :return: 对比损失（标量）
        """
        n = labels.shape[0] # 样本数
        # 生成相似度矩阵
        similarity_matrix = torch.div(torch.matmul(features, features.T),
                                    self.temperature)

        # 减去相似度矩阵每一行的最大值，防止后面计算exp上溢
        similarity_matrix_max, _ = torch.max(similarity_matrix, dim=1,  keepdim=True)
        similarity_matrix = similarity_matrix - similarity_matrix_max

        # 生成掩码矩阵，表示相似度矩阵中哪些元素是同类的（不包括本身）
        labels = labels.reshape(-1, 1) # 增加一个维度，用于后面转置计算
        mask_matrix = torch.eq(labels, labels.T).float()
        diag_zeros = torch.ones(n, n) - torch.eye(n, n) # 生成对角为0，其他元素为1的矩阵
        mask_matrix = mask_matrix * diag_zeros

        # 计算每个相似度的exp，并去掉本身的相似度（不需要计算）
        exp_similarity_matrix = torch.exp(similarity_matrix) * diag_zeros

        # 计算对比损失公式中第一个求和里面的项
        log_probs = - similarity_matrix + \
                    torch.log(exp_similarity_matrix.sum(1,keepdims=True))

        # 求对比损失公式的第一个求和
        mean_log_probs = (log_probs * mask_matrix).sum(1) / mask_matrix.sum(1)

        # 计算对比损失公式的第二个求和，即计算对比损失
        loss = mean_log_probs.mean()

        return loss



if __name__ == '__main__':
    representations = torch.tensor([[1, 2, 3], [1.2, 2.2, 3.3],
                                    [1.3, 2.3, 4.3], [1.5, 2.6, 3.9],
                                    [5.1, 2.1, 3.4]])
    representations = F.normalize(representations, dim=1)
    print(representations)
    labels = torch.tensor([1, 0, 0, 1, 1])
    labels = labels.repeat(2, 1)
    print(labels)
