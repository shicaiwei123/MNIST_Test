import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as tm

# os.environ['CUDA_VISIBLE_DEVICES']='1'   # 指定用于训练的GPU，否则默认是第0块GPU

# 训练参数

# Hyper Parameters

EPOCH = 5  # 训练的轮数,这里只迭代一五轮
LR = 0.001  # 学习率
momentum = 0.90
batch_size = 32  # 每次训练的时候,放入多少张图片或者是样本


# 读取数据
def get_data_loader():
    data_train = datasets.MNIST(root='./data/',  # 数据集的目录
                                train=True,  # 用于训练
                                transform=transforms.ToTensor(),  # 转换成张量
                                download=True)  # 是否从网络上下载,如果自己已经下载好了可以不用

    data_test = datasets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

    print(data_train.data.size())  # (60000, 28, 28)
    print(data_train.targets.size())  # (60000)
    # 加载数据

    train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)

    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True,
                                               sampler=train_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(data_test)

    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True,
                                              sampler=test_sampler)
    return train_loader, test_loader


# 定义网络
class C_CNN_Net(torch.nn.Module):
    def __init__(self):
        super(C_CNN_Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)  # 将输出的特征图展开成一维向量
        x = self.dense(x)
        return x


def train(epoch, model, train_loader, optimizer):  # 定义每个epoch的训练细节
    running_loss = 0
    running_correct = 0
    train_loss = 0
    model.train()  # 设置为trainning模式
    batch_num = 0
    for epoch in range(1, epoch + 1):  # 以epoch为单位进行循环
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_num += 1
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()  # 数据迁移到GPU上
            optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度

            output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
            loss = cost(output, target)  # 计算损失函数
            train_loss += loss.item()
            loss.backward()  # 反向传播求出输出到每一个节点的梯度
            optimizer.step()  # 根据输出到每一个节点的梯度,优化更新参数
            if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        print("Train Epoch", epoch, "loss", train_loss / batch_num)
        train_loss = 0
    torch.save(model, 'minist_cnn.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(model.state_dict(), 'minist_cnn_params.pkl')  # 只保存神经网络的模型参数


def test(test_loader):
    model = torch.load('minist_cnn.pkl')
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    true_wrong = 0
    true_right = 0
    false_wrong = 0
    false_right = 0

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)  # 数据是按照batch的形式喂入的,然后是这里的输出是全连接层的输出结果
        test_loss += (
            cost(output, target)).item()  # sum up batch 求loss 把所有loss值进行累加.一般分类器会有softmax,但是这里没有是因为集成在这个损失函数之中了
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability #输出的结果虽然会过sofamax转成概率值,但是相对打大小是不变的,所以直接输出取最大值对应的索引就是分类结果
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        compare_result = pred.eq(target.data.view_as(pred)).cpu()  # 判断输出和目标数据是不是一样,然后将比较结果转到cpu上
        # target=target.numpy()
        target = target.cpu()
        compare_result = np.array(compare_result)
        for i in range(len(compare_result)):
            if compare_result[i]:
                if target[i] == 1:
                    true_right += 1
                else:
                    false_right += 1
            else:
                if target[i] == 1:
                    true_wrong += 1
                else:
                    false_wrong += 1

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def init_distributed_mode(args):
    '''initilize DDP
    '''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}",
        flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument('--local_rank', type=int, help='local rank, will passed by ddp')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    init_distributed_mode(args)

    torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")



    # # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    # # torch.cuda.set_device(local_rank)
    # #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    # dist.init_process_group(backend='nccl')
    #
    # print(type(args.local_rank))
    # local_rank = args.local_rank
    device = torch.device("cuda", int(args.local_rank))
    #
    # print('world_size', torch.distributed.get_world_size())

    train_loader, test_loader = get_data_loader()

    # 实例化模型并且打印
    model = C_CNN_Net()
    model = model.to(device)
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    # distiller = torch.nn.SyncBatchNorm.convert_sync_batchnorm(distiller)
    # print(device, local_rank, torch.cuda.is_available())
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    # 训练和测试
    cost = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)  # 初始化优化器 model.train()

    log_interval = 10  # 每10个batch输出一次信息
    train(EPOCH, model, train_loader, optimizer)
