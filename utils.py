#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:33:05 2019

@author: jiaruizheng
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, size_average=True, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):

        logpt = F.cross_entropy(input, target, weight=self.weight)
        pt = Variable(logpt.data.exp())

        # compute the loss
        loss = (1 - pt) ** self.gamma * logpt
        return loss


def arr2df(arr1,arr2,arr3):
    df1 = pd.DataFrame(arr1)
    df2 = pd.DataFrame(arr2)
    df3 = pd.DataFrame(arr3)
    concat_df = pd.concat([df1,df2,df3],axis=1)
    return concat_df


def get_label_weight(train_table):
    label_list = []
    for i in train_table.itertuples(index=False):
        label_list.append(i[0])
    train = pd.DataFrame(label_list)
    N = train.shape[0]
    weight_list = []
    for i in train.columns:
        num_0 = len(train[train[i] == 0])
        num_1 = len(train[train[i] == 1])
        weight = [1.0*N/num_0, 1.0*N/num_1]
        weight_list.append(weight)
    return weight_list


def Loss(probs,labels,weight_list, use_focal=True):
    loss = 0.0
    probs = probs.view(labels.size()[0], -1)
    if use_focal:
        for i in range(labels.size()[1]):
            weight_i = torch.FloatTensor([0.25, 0.75]).to(device)
            loss += FocalLoss(weight=weight_i)(probs[:, i*2:(i+1)*2], labels[:, i])
        return loss
    else:
        for i in range(labels.size()[1]):
            weight_i = torch.FloatTensor(weight_list[i]).to(device)
            loss += F.cross_entropy(probs[:, i*2:(i+1)*2], labels[:, i], weight=weight_i)
        return loss


# Help function to generate tags from predicion outputs
def Pred2tag(arr, inputs, num_labels, label_names):
    pred_tags = []
    for i in range(inputs.size()[0]):
        arr_i = arr[i*num_labels:i*num_labels+num_labels]
        index = np.where(arr_i == 1)
        index = index[0].tolist()
        pred_tag = [label_names[j] for j in index]
        pred_tags.append(pred_tag)
    return pred_tags


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def Evaluate(preds, labels):
    preds = (preds.cpu().detach().numpy()).astype(np.int8)
    labels = (labels.view(-1).cpu().detach().numpy()).astype(np.int8)
    tp = np.sum(preds*labels, axis=0)
    fp = np.sum((1 - preds) * labels, axis=0)
    fn = np.sum(preds * (1 - labels), axis=0)
    return tp, fp, fn


