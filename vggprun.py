import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from model.vgg import vgg
from utils import *
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.3,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='model_result/vgg19_best_ckpt.t7', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='model_result/vgg19_prun_best_ckpt.t7', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = vgg()
print(model)
model.to(device)
# if args.cuda:
#     model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)# ,map_location='cpu'
        model.load_state_dict(checkpoint['net'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model_):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    ])
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    model_.eval()
    correct = 0
    totalnum = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model_(data)
            _, pred = torch.max(output.data, 1)
            totalnum += target.size(0)
            correct += pred.eq(target).sum().item()
            # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            progress_bar(batch_idx, len(test_loader), ' ||| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / totalnum, correct, totalnum))
        print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
            correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))


def generate_prunnet(model_):
    total = 0
    for m in model_.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model_.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]
    thre = thre.type(torch.cuda.FloatTensor)
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model_.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    print(cfg)

    pruned_ratio = pruned/total

    print('pruned_ratio: %s' % pruned_ratio)
    print('Pre-processing Successful!')
    return cfg, cfg_mask


def generate_newmodel(model_, cfg, cfg_mask):
    # Make real prune
    newmodel = vgg(cfg=cfg)
    newmodel.to(device)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model_.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            w = m0.weight.data[:, idx0, :, :].clone()
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
            # m1.bias.data = m0.bias.data[idx1].clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[:, idx0].clone()
    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)
    print(newmodel)
    test(newmodel)


if __name__ == '__main__':
    # #test model before pruning
    test(model)

    cfg, cfg_mask = generate_prunnet(model)
    generate_newmodel(model, cfg, cfg_mask)
    print('Finish pruning .....')
