# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# from model.resnet import *
from model.vgg import *
from model.preresnet import preresnet
from utils import *
import time
import numpy as np
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='knowledge distillation CIFAR training')
parser.add_argument('--model', type=str, default='vgg', help='CNN architecture')
parser.add_argument('--sr', default=True, help='Sparsity')
parser.add_argument('--s', type=float, default='1e-5', help='sr param')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='training batchsize')
parser.add_argument('--epochs', type=int, default=160, help='training epoches')
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
args = parser.parse_args()
# Set the logger
set_logger(os.path.join('model_result', 'train.log'))
best_acc = 0

transform_train = transforms.Compose([
    # transforms.RandomResizedCrop(32, interpolation=2),
    transforms.RandomCrop(32, padding=4),
    # transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
])

logging.info("Loading the datasets...")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

# compute mean and std
# trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# print(trainset.train_data.shape)
# print(np.mean(trainset.train_data, axis=(0, 1, 2)) / 255)
# print(np.std(trainset.train_data, axis=(0, 1, 2)) / 255)
# print(aaa)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logging.info("Train Net:{}".format(args.model))
if args.model == 'preresnet':
    net = preresnet(depth=40)
elif args.model == 'vgg':
    net = vgg()
elif args.model == 'refine':
    refine_model = 'model_result/vgg19_prun_best_ckpt.t7'
    checkpoint = torch.load(refine_model)
    net = vgg(cfg=checkpoint['cfg'])
    net.load_state_dict(checkpoint['state_dict'])
net.to(device)

# if device == 'cuda':
#     #model = nn.DataParallel(net, device_ids=[0, 1])  # multi-GPU
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.t7')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
# criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 130], gamma=0.2)
# # 余弦退火
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1


# Training
def train(epoch):
    lr = optimizer.param_groups[0]['lr']
    print('\nEpoch: %d LR: %f' % (epoch, lr))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        # loss = criterion(outputs, targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        loss.backward()

        if args.sr:
            updateBN()
        optimizer.step()

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx+1), 100.*correct/total, correct, total))
    train_loss = train_loss / (batch_idx+1)
    print('AVG Train loss :%.4f' % train_loss)
    train_acc = 100.0*correct/total
    return train_loss, train_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # loss = criterion(outputs, targets)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx+1), 100.*correct/total, correct, total))

    test_loss = test_loss / (batch_idx+1)
    # Save model_result.
    test_acc = 100.*correct / total
    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('model_result'):
            os.mkdir('model_result')
        name = 'vgg19_best_ckpt.t7'
        save_path = os.path.join('model_result', name)
        torch.save(state, save_path)
        best_acc = test_acc
    print('best_acc :%.4f' % best_acc)
    return test_loss, test_acc


def figure(trainloss, testloss, trainacc, testacc):
    assert len(trainloss) == len(testloss)
    epoch = len(trainloss)
    x = range(0, epoch)
    y1 = trainloss
    y2 = testloss
    y3 = trainacc
    y4 = testacc

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.title('Loss VS epoch')
    plt.plot(x, y1, color='blue', label='train-loss')
    plt.plot(x, y2, color='red', label='test-loss')
    plt.legend()# 显示图例
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.title('Acc VS epoch')
    plt.plot(x, y3, color='blue', label='train-acc')
    plt.plot(x, y4, color='red', label='test-acc')
    plt.legend()# 显示图例
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.savefig('figure/teacher_vgg_loss_acc.png')


if __name__ == '__main__':
    trainloss = []
    trainacc = []
    testloss = []
    testacc = []
    for epoch in range(1, args.epochs):
        train_scheduler.step(epoch)
        train_loss, train_acc = train(epoch)
        trainloss.append(train_loss)
        trainacc.append(train_acc)

        test_loss, test_acc = test(epoch)
        testloss.append(test_loss)
        testacc.append(test_acc)
        logging.info("train_loss:{} train acc:{} test_loss:{} test_acc:{}"
                     .format(train_loss, train_acc, test_loss, test_acc))
    logging.info("teacher best acc:{}".format(best_acc))
    figure(trainloss, testloss, trainacc, testacc)
    logging.info("Finished Training...")

