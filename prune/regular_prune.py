#coding=utf-8
import os
import argparse
import torch
import torch.nn as nn
from torchvision.models import AlexNet
from torchviz import make_dot
import tensorwatch as tw
from torch.autograd import Variable
from torchvision import datasets, transforms
# import util
from models.model import create_model, load_model
import numpy as np
from models import *
from opts import opts
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', default='../data',
                    help='dataset path')# test usage data
parser.add_argument('--percent', type=float, default=0.4,
                    help='nin:0.5') # percent of pruning
#TODO now using regular
parser.add_argument('--normal_regular', type=int, default=1,
                    help='--normal_regular_flag (default: normal)') # regualr|norm, 1=regualr
parser.add_argument('--layers', type=int, default=100,
                    help='layers (default: 9)') # model layer
parser.add_argument('--model', default='../model/all_dla34.pth', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)') # model input
parser.add_argument('--save', default='../model/prune.pth', type=str, metavar='PATH',
                    help='path to save prune model (default: none)') # model save
args = parser.parse_args()
base_number = args.normal_regular
layers = args.layers
print(args)

#check
if base_number <= 0:
    print('\r\n!base_number is error!\r\n')
    base_number = 1

# model loading and checking and vis
opt = opts().init()
model = create_model(opt.arch, opt.heads, opt.head_conv) # dla_34, from opt.task, -1 default
print("=> loading checkpoint '{}'".format(args.model))
model = load_model(model, opt.load_model)
print('***********************************Model**************************************')
print(model)
# img = tw.draw_model(model,[1, 3, 1088, 608])
# img.save(r'dla_structure.jpg')
# x = torch.randn(1, 3, 1088, 608)
# y=model(x)
# g = make_dot(y[0]['reg'])
# g.render('dla34_structure', view=False)
# macs, params = profile(model, inputs=(input, ))
# print(macs/(10**6),params/(10**6)," M")
print('parameters(M):', (sum(param.numel() for param in model.parameters()))/10**6)

total = 0
i = 0
for m in model.modules():
    # calculate the channel num of BN layer
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            total += m.weight.data.shape[0]

# calculate the num of all weights
bn = torch.zeros(total)
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
# rank as the weight, ↑
y, j = torch.sort(bn)
thre_index = int(total * args.percent)
if thre_index == total:
    thre_index = total - 1
# set the threshold of weight
thre_0 = y[thre_index]

#********************************PRE-PRUNE*********************************
pruned = 0
cfg_0 = []
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone()
            # channels needed
            mask = weight_copy.abs().gt(thre_0).float()
            remain_channels = torch.sum(mask)
            # if all channels in this layer is pruned
            if remain_channels == 0:
                print('\r\n!please turn down the prune_ratio!\r\n')
                remain_channels = 1
                mask[int(torch.argmax(weight_copy))]=1

            # ******************Regular Prune, make the channel num looks regular******************
            v = 0
            n = 1
            if remain_channels % base_number != 0:
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)

                    y, j = torch.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()
            # the num of pruned channels
            pruned = pruned + mask.shape[0] - torch.sum(mask) # calcualte channel left
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask) # pre-prune batch_size channel
            cfg_0.append(mask.shape[0]) # original channel shapes.
            cfg.append(int(remain_channels)) # pruned channel shapes
            cfg_mask.append(mask.clone())
            print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                  format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
pruned_ratio = float(pruned/total)
print('\r\n!pre-pruned finished!')
print('total_pruned_ratio: ', pruned_ratio)

#********************************PRUNE*********************************
newmodel = create_model(opt.arch, opt.heads, opt.head_conv)
# nin.Net(cfg)  # TODO must modify the original model structure
newmodel.cuda()
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
i = 0
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            # np.squeeze delete single dim in shape.
            # np.argwhere(a) return index of non-zero elm.
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,)) # for matching bn shape
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            # init: end = start, update: start = end, end = stack[next], serve as conv in_c and out_c
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Conv2d):
        if i < layers - 1:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            tmp = m0.weight
            w = m0.weight.data[:, idx0.tolist(), :, :].clone()
            print(w.shape)
            m1.weight.data = w[idx1.tolist(), :, :, :].clone()
            if m0.bias is not None:
                m1.bias.data = m0.bias.data[idx1].clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
            if m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.Linear):
        # noted: conv Tensor's dim is [n, c, w, h]
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        if m0.bias is not None:
            m1.bias.data = m0.bias.data.clone()

#******************************model save*********************************
print('**********Save Pruned Model*********')
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)
print('**********Save Model Success*********\r\n')
