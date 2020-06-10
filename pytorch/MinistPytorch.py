import torch
import torch.nn as nn

from config import *
from models.model_net import C_CNN_Net
import time
from utils import get_cafar10_loader, get_mnist_loader, train_base, test_base
from torchtoolbox.tools import mixup_criterion, mixup_data


def train_mnist():  # 定义每个epoch的训练细节

    # 定义数据

    train_loader = get_mnist_loader(train=True, batch_size=args.batch_size)
    test_loader = get_mnist_loader(train=False, batch_size=args.batch_size)

    # 定义模型
    # 实例化模型并且打印
    model = C_CNN_Net()
    print(model)

    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    # 定义优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)

    # 开始训练

    args.retrain = True
    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader,
               args=args)


def test_mnist(model_path, cost, test_loader):
    '''

    :param model_path: 要测试的模型的参数保存位置
    :param cost: 损失函数
    :param test_loader:测试数据集
    :return:
    '''
    # 模型加载
    state_read = torch.load(model_path)
    model = C_CNN_Net()
    model = model.load_state_dict(state_read['model_sate'])

    test_base(model, cost=cost, test_loader=test_loader)


if __name__ == '__main__':
    train_mnist()
# test()
