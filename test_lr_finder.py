# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from model.resnet import *
from model.xception import *
from utils import *
from lr_finder import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='knowledge distillation CIFAR training')
parser.add_argument('--model', type=str, default='xception', help='CNN architecture')
parser.add_argument('--base_lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--max_lr', type=float, default=1, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='training batchsize')
parser.add_argument('-num_iter', type=int, default=300, help='num of iteration')
parser.add_argument('--epochs', type=int, default=150, help='training epoches')
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, interpolation=2),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if args.model == 'xception':
    net = xception()
net.to(device)

# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

# lr_finder = LRFinder(net, optimizer, criterion, device=device)
# lr_finder.range_test(trainloader, end_lr=1, num_iter=200)
# lr_finder.plot(log_lr=False)
# print('Finished Training')


def train_findlr(beta=0.98):
    net.train()
    num = len(trainloader)-1
    mult = (args.max_lr / args.base_lr) ** (1/num)
    lr = args.base_lr
    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for batch_index, (inputs, targets) in enumerate(trainloader):
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        print('Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            iter_num=batch_num,
            trained_samples=batch_index * args.bs + len(inputs),
            total_samples=len(trainloader.dataset),
        ))
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


def figurepng(log_lrs, losses):
    plt.figure()
    plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.plot(log_lrs, losses)
    # plt.show()
    plt.savefig('figure/result_lr.png')


if __name__ == '__main__':
    # mp.set_start_method('forkserver')
    log_lrs, losses = train_findlr(beta=0.98)
    figurepng(log_lrs, losses)
    print('Finished Training')
