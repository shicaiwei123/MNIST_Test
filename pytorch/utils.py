import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torchtoolbox.tools import mixup_criterion, mixup_data
import time
import csv
import torch.optim as optim
import os


def get_mnist_loader(train=True, batch_size=256):
    """
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: MNIST loader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    data_set = torchvision.datasets.MNIST(root='./data', train=train,
                                          download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def get_cafar10_loader(train=True, batch_size=1):
    """
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: MNIST loader
    """
    data_set = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=True, transform=transforms.ToTensor())

    # mean, std = get_mean_std(data_set)
    # print("mean",mean,"std",std)
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.49, 0.48, 0.44,), std=(0.24, 0.24, 0.26))])
    # transform=None
    data_set = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                         shuffle=train, num_workers=4)

    return loader


def calc_accuracy(model, loader, verbose=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)
        outputs_full.append(outputs_batch)
        labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    return accuracy


def train_base(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model: 要训练的模型
    :param cost: 损失函数
    :param optimizer: 优化器
    :param train_loader:  测试数据装载
    :param test_loader:  训练数据装载
    :param args: 配置参数
    :return:
    '''

    # 打印训练参数
    print(args)

    # 初始化,打开定时器,创建保存位置
    start = time.time()

    models_dir = args.model_root + args.model_name
    log_dir = args.log_root + args.log_name

    # 保存argv参数
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # 判断是否使用余弦衰减
    if args.lrcos:
        print("lrcos is using")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=0)

    # 训练初始化
    epoch_num = args.train_epoch  # 训练多少个epoch
    log_interval = args.log_interval  # 每隔多少个batch打印一次状态
    save_interval = args.save_interval  # 每隔多少个epoch 保存一次数据

    batch_num = 0
    train_loss = 0
    log_list = []  # 需要保存的log数据列表

    epoch = 0
    accuracy_best=0

    # 如果是重新训练的过程,那么就读取之前训练的状态
    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # 训练
    while epoch < epoch_num:  # 以epoch为单位进行循环
        for batch_idx, (data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                print("mixup is using")
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度

            output = model(data)  # 把数据输入网络并得到输出，即进行前向传播

            if args.mixup:
                loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()  # 反向传播求出输出到每一个节点的梯度
            optimizer.step()  # 根据输出到每一个节点的梯度,优化更新参数```````````````````````````````````````````````````
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # 测试模型
        accuracy_test = calc_accuracy(model, loader=test_loader)
        if accuracy_test>accuracy_best:
            accuracy_best=accuracy_test
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch, train_loss / len(train_loader),
                                                                          accuracy_test, accuracy_best))
        train_loss = 0

        if args.lrcos:
            scheduler.step(epoch=epoch)

        # 保存模型和优化器参数
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            torch.save(train_state, models_dir)

        # 保存log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def test_base(model, cost, test_loader):
    '''
    :param model: 要测试的带参数模型
    :param cost: 损失函数
    :param test_loader: 测试集
    :return:
    '''

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
