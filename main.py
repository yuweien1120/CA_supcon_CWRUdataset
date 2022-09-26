import os
import random  # to set the python random seed
import numpy  # to set the numpy random seed
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

"""自定义的Dataloader，网络和损失函数"""
from cwru_processing import CWRU_DataLoader
from SupConLoss import SupConLoss
from Model.ResNet1d import create_ResNetCWRU
from Model.dot_product_classifier import DotProduct_Classifier

"""记录参数并上传wandb"""
from utils.metric import AverageMeter, accuracy
import wandb
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

def test(model_dict, device, lossfn_dict, test_loader):
    """
    上传模型在测试集上的测试指标到wandb网页，
    测试指标包括：测试标签位于模型输出前1的正确率，测试标签位于模型输出前5的正确率，测试的损失值
    :param model_dict: 网络模型（字典索引）
    :param device: 训练使用的设备，cuda或cpu
    :param lossfn_dict: 损失函数（字典索引）
    :param test_loader: 测试训练集
    :return:
    """
    model_dict['Backbone'].eval()
    model_dict['Classifier'].eval()
    test_loss = AverageMeter()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], 1, data.shape[1])
            feature = model_dict['Backbone'](data)
            output = model_dict['Classifier'](feature)
            loss = lossfn_dict['classifier_loss'](output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            test_top1.update(prec1.item(), n=target.size(0))
            test_top5.update(prec5.item(), n=target.size(0))
            test_loss.update(loss.item(), n=target.size(0))

    wandb.log({
        "Test top 1 Acc": test_top1.avg,
        "Test top5 Acc": test_top5.avg,
        "Test Loss": test_loss.avg})
    # return test_top1.avg 如果是验证集，则可以返回acc

def train(model_dict, device, train_loader, optimizer, lossfn_dict, epoch, epochs, lamda):
    """
    对模型进行一轮训练，并打印和上传相关的训练指标
    训练指标包括：训练标签位于模型输出前1的正确率，训练标签位于模型输出前5的正确率，训练的损失值
    :param model_dict: 多个网络模型（字典索引）
    :param device: 训练使用的设备，cuda或cpu
    :param train_loader: 训练集
    :param optimizer: 训练优化器
    :param lossfn_dict: 特征损失函数和分类损失函数（字典索引）
    :param epoch: 训练轮数
    :param epochs: 训练总轮数
    :param lamda: 对比损失的系数
    :return: 训练标签位于模型输出前1的正确率
    """
    model_dict['Backbone'].train()
    model_dict['Classifier'].train()
    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    # 用原始数据经过Backbone和分类器计算分类损失，数据增强的数据经过Backbone计算对比损失，最后综合更新参数
    for batch_idx, (data, target, data_transformed1, data_transformed2) in enumerate(train_loader):
        data, target, data_transformed1, data_transformed2 = data.to(device), target.to(device), \
                                                            data_transformed1.to(device), \
                                                            data_transformed2.to(device)

        optimizer.zero_grad()

        data = data.reshape(data.shape[0], 1, data.shape[1])
        data_transformed1 = data_transformed1.reshape(data_transformed1.shape[0], 1, data_transformed1.shape[1])
        data_transformed2 = data_transformed2.reshape(data_transformed2.shape[0], 1, data_transformed2.shape[1])

        # 利用数据增强后的数据生成两组特征，组合成一组特征，计算对比损失
        feature1 = model_dict['Backbone'](data_transformed1)
        feature2 = model_dict['Backbone'](data_transformed2)
        # 将特征归一化，防止后面计算对比损失出现数值不稳定
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        # 合并feature1和feature2
        feature_supcon = torch.vstack((feature1, feature2))
        # 将标签复制一遍，相当于合并feature1和2的标签
        sup_target = target.repeat(2)
        # 计算对比损失
        sup_loss = lossfn_dict['feature_loss'](feature_supcon, sup_target)

        # 利用原始数据经过网络生成分类特征
        feature = model_dict['Backbone'](data)
        output = model_dict['Classifier'](feature)
        # 计算分类损失
        classifier_loss = lossfn_dict['classifier_loss'](output, target)

        # 综合计算损失
        total_loss = classifier_loss + lamda * sup_loss

        with torch.no_grad():
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            train_loss.update(total_loss.item(), n=target.size(0))
            train_top1.update(prec1.item(), n=target.size(0))
            train_top5.update(prec5.item(), n=target.size(0))

        total_loss.backward()
        optimizer.step()
        # 判断loss当中是不是有元素非空，如果有就终止训练，并打印梯度爆炸
        if np.any(np.isnan(total_loss.item())):
            print("Gradient Explore")
            break
        # 每训练20个小批量样本就打印一次训练信息
        if batch_idx % 20 == 0:
            print('Epoch: [%d|%d] Step:[%d|%d], LOSS: %.5f' %
                (epoch, epochs, batch_idx + 1, len(train_loader), total_loss.item()))

    wandb.log({
        "Train top 1 Acc": train_top1.avg,
        "Train top5 Acc": train_top5.avg,
        "Train Loss": train_loss.avg})
    return train_top1.avg

def main(config):
    # 配置训练模型时使用的设备（cpu/cuda）
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 如果使用cuda则修改线程数和允许加载数据到固定内存
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # 设置随机数种子，保证结果的可复现性
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    numpy.random.seed(config.seed)  # numpy random seed
    # 固定返回的卷积算法，保证结果的一致性
    torch.backends.cudnn.deterministic = True

    # 读取数据集
    path = r'Data\48k_DE\HP3'
    train_loader, valid_loader, test_loader = CWRU_DataLoader(d_path=path,
                                                            length=400,
                                                            step_size=200,
                                                            train_number=1800,
                                                            test_valid_number=300,
                                                            batch_size=60,
                                                            normal=True,
                                                            IB_rate=[10, 1],
                                                            transforms_name=config.transforms)

    # 定义使用的网络模型（残差网络和分类器）
    model_dict = {}
    model_dict['Backbone'] = eval(config.model_names['Backbone'])\
                                (**config.model_params['Backbone']).to(device)
    model_dict['Classifier'] = eval(config.model_names['Classifier'])\
                                (**config.model_params['Classifier']).to(device)

    # 定义损失函数
    lossfn_dict = {}
    lossfn_dict['feature_loss'] = eval(config.lossfn_names['feature_loss'])\
                                        (**config.lossfn_params['feature_loss']).to(device)
    lossfn_dict['classifier_loss'] = eval(config.lossfn_names['classifier_loss'])\
                                        (**config.lossfn_params['classifier_loss']).to(device)

    # 定义优化器
    optim_params_list = [{'params': model_dict['Backbone'].parameters(),
                          **config.optim_params['Backbone']},
                        {'params': model_dict['Classifier'].parameters(),
                        **config.optim_params['Classifier']}]
    optimizer = optim.Adam(optim_params_list)

    # 追踪模型参数并上传wandb
    wandb.watch(model_dict['Backbone'], log="all")
    wandb.watch(model_dict['Classifier'], log="all")

    # 设置期望正确率
    max_acc = 0
    acc = 0

    # 定义存储模型参数的文件名
    model_paths = {'Backbone': config.datasets + '_' + config.model_names['Backbone'] + '.pth',
                'Classifier': config.datasets + '_' + config.model_names['Classifier'] + '.pth'}

    # 训练模型
    for epoch in range(1, config.epochs + 1):
        acc = train(model_dict, device, train_loader, optimizer, lossfn_dict, epoch, config.epochs, config.lamda)

        # 如果acc为非数，则终止训练
        if np.any(np.isnan(acc)):
            print("NaN")
            break

        # 当训练正确率超出期望值，存储模型参数
        if acc > max_acc:
            torch.save(model_dict['Backbone'].state_dict(),
                    os.path.join(wandb.run.dir, model_paths['Backbone']))
            torch.save(model_dict['Classifier'].state_dict(),
                    os.path.join(wandb.run.dir, model_paths['Classifier']))


        # 每训练完一轮，测试模型的指标
        test(model_dict, device, lossfn_dict, test_loader)

    # 如果正确率非空，则保存模型到wandb
    if not np.any(np.isnan(acc)):
        wandb.save('*.pth')

if __name__ == '__main__':
    """定义wandb上传项目名"""
    wandb.init(project="CA_Supcon", name="IB=10:1")
    wandb.watch_called = False

    """定义上传的超参数"""
    config = wandb.config
    """数据集及其预处理"""
    config.datasets = 'CWRU(48k_DE_HP3)'  # 数据集
    config.transforms = ("Add_Gaussian", "Random_Scale", "Mask_Noise", "Translation")  # 数据增强
    config.batch_size = 60  # 批量样本数
    config.sampler = "ClassAwareSampler"  # 训练集采样器
    config.num_sampler_cls = 6  # 采样器重复采样某一类别样本次数
    config.test_batch_size = 60  # 测试批量样本数

    """网络模型的参数"""
    config.model_names = {'Backbone': 'create_ResNetCWRU', 'Classifier': 'DotProduct_Classifier'}  # 网络模型
    ResNetCWRU_params = {'seq_len': 400, 'num_blocks': 2, 'planes': [10, 10, 10],
                                'kernel_size': 10, 'pool_size': 2, 'linear_plane': 100}
    DotProduct_Classifier_params = {'linear_plane': 100, 'num_classes': 10}
    config.model_params = {'Backbone': ResNetCWRU_params,
                        'Classifier': DotProduct_Classifier_params}  # 网络模型的相关参数

    """损失函数的参数"""
    config.lossfn_names = {'feature_loss': 'SupConLoss',
                        'classifier_loss': 'nn.CrossEntropyLoss'}  # 损失函数
    SupConLoss_params = {'temperature': 0.1}
    CrossEntropyLoss_params = {}
    config.lossfn_params = {'feature_loss': SupConLoss_params,
                            'classifier_loss': CrossEntropyLoss_params}  # 损失函数的参数

    """训练的相关参数"""
    config.epochs = 50  # 训练轮数
    config.lr = 0.0003  # 学习率
    config.beta = (0.9, 0.99)  # 平滑常数
    config.weight_decay = 0.0003  # 权重衰减
    config.optimizer = 'Adam'
    ResNetCWRU_optim_params = {'lr': config.lr, 'beta': config.beta,
                                'weight_decay': config.weight_decay}
    DotProduct_Classifier_optim_params = {'lr': config.lr, 'beta': config.beta,
                                        'weight_decay': config.weight_decay}
    config.optim_params = {'Backbone': ResNetCWRU_optim_params,
                        'Classifier': DotProduct_Classifier_optim_params}
    config.lamda = 1
    config.no_cuda = False  # 不使用cuda（T/F）

    config.seed = 0  # 随机数种子

    main(config)
