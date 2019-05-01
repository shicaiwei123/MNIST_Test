import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# 训练参数

# Hyper Parameters

EPOCH = 15  # 训练的轮数,这里只迭代一轮
LR = 0.001  # 学习率
batch_size = 256  # 每次训练的时候,放入多少张图片或者是样本

# 读取数据
data_train = datasets.MNIST(root='./data/',  # 数据集的目录
                            train=True,  # 用于训练
                            transform=transforms.ToTensor(),  # 转换成张量
                            download=False)  # 是否从网络上下载,如果自己已经下载好了可以不用

data_test = datasets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor())

print(data_train.train_data.size())  # (60000, 28, 28)
print(data_train.train_labels.size())  # (60000)
# 加载数据

train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                           batch_size=batch_size,
                                           shuffle=True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                          batch_size=batch_size,
                                          shuffle=True)


# 定义网络
# 基于参数和传播算法
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
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


# 实例化模型并且打印
model = C_CNN_Net()
print(model)

# 如果有GPU
if torch.cuda.is_available():
    model.cuda()  # 将所有的模型参数移动到GPU上

# 损失函数
cost = torch.nn.CrossEntropyLoss()

# 优化器
optimzer = torch.optim.Adam(model.parameters())

# 训练和测试
cost = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 初始化优化器 model.train()

log_interval = 10  # 每10个batch输出一次信息


def train(epoch):  # 定义每个epoch的训练细节
    running_loss = 0
    running_correct = 0
    model.train()  # 设置为trainning模式
    for epoch in range(1, epoch + 1):  # 以epoch为单位进行循环
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)  # 把数据转换成Variable
            optimizer.zero_grad()  # 优化器梯度初始化为零
            output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
            loss = cost(output, target)  # 计算损失函数
            loss.backward()  # 反向传播梯度
            optimizer.step()  # 结束一次前传+反传之后，更新优化器参数
            if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model, 'minist_cnn.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(model.state_dict(), 'minist_cnn_params.pkl')  # 只保存神经网络的模型参数


def test():
    model = torch.load('minist_cnn.pkl')
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += (cost(output, target)).item()  # sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train(EPOCH)
test()
