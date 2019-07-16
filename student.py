# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from model.senet import *
from model.resnet import *
from model.squeezenet import *
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import argparse
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='knowledge distillation CIFAR training')
parser.add_argument('--model', type=str, default='squee', help='CNN architecture')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='training batchsize')
parser.add_argument('--epochs', type=int, default=100, help='training epoches')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
args = parser.parse_args()
# Set the logger
set_logger(os.path.join('model_result', 'train_student.log'))
best_acc = 0

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, interpolation=2),
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
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

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


"""
netT = ResNet18()
netT.to(device)
netT.eval()

path = 'saved_model/'
model_path = os.path.join(path, 'PublicTest_model.t7')
checkpoint = torch.load(model_path)#,map_location='cpu'
net.load_state_dict(checkpoint['net'])

# netT = torch.load('teacher.pkl')
# netT.to(device)
soft_target = torch.tensor([]).to(device)
with torch.no_grad():
    for data in trainloader:
        images, _ = data
        images = images.to(device)
        outputs = netT(images)
        soft_target = torch.cat((soft_target, outputs), 0)
soft_target.to("cpu")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                          shuffle=False, num_workers=2)

with torch.no_grad():
    for data in trainloader:
        images, labels = data

softset = torch.utils.data.TensorDataset(images, labels, soft_target)
"""
logging.info("Train Net:{}".format(args.model))
if args.model == 'squee':
    net = squeezenet()
net.to(device)
if device == 'cuda':
    # model = nn.DataParallel(model, device_ids=[0, 1, 2])  # multi-GPU
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Load teacher model
logging.info("Load netT model ......")
netT = ResNet50()
path = 'model_result/'
model_path = os.path.join(path, 'resnet50_best_0.708_200.t7')
checkpoint = torch.load(model_path)# ,map_location='cpu'
netT.load_state_dict(checkpoint['net'])
netT.to(device)
netT.eval()


def train(epoch):
    lr = optimizer.param_groups[0]['lr']
    print('\nEpoch: %d LR: %f' % (epoch, lr))
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    alpha = 0.95
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        soft_target = netT(inputs)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss1 = criterion(outputs, targets)

        T = 10
        outputs_S = F.log_softmax(outputs/T, dim=1)
        outputs_T = F.softmax(soft_target/T, dim=1)
        loss2 = criterion2(outputs_S, outputs_T) * T * T

        loss = loss1 * (1 - alpha) + loss2 * alpha

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx+1), 100.*correct/total, correct, total))
    train_loss = train_loss / (batch_idx + 1)
    print('AVG Train loss :%.4f' % train_loss)
    train_acc = 100.0 * correct / total
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
            loss = criterion(outputs, targets)
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
        torch.save(state, 'model_result/squeezenet_best_student.t7')
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

    plt.figure(figsize=(16, 6))
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
    plt.savefig('figure/student_loss_acc.png')


if __name__ == '__main__':
    # mp.set_start_method('forkserver')
    trainloss = []
    trainacc = []
    testloss = []
    testacc = []
    for epoch in range(args.epochs):
        # if epoch > 60:
        #     optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
        # if epoch > 120:
        #     optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9, weight_decay=5e-4)
        # if epoch > 160:
        #     optimizer = optim.SGD(net.parameters(), lr=0.0008, momentum=0.9, weight_decay=5e-4)
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

"""
x = torch.tensor([[ 0.2979,  0.0655, -0.0312,  0.0616,  0.0830, 
                   -0.1206, -0.2084, -0.0345,  0.2106, -0.0558]])
y = torch.tensor([5])
print(torch.log(torch.sum(torch.exp(x))) - x[0, y])

criterion = nn.CrossEntropyLoss()
print(criterion(x, y))
"""