#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:14:41 2019

@author: jiaruizheng
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloader import *
from models import *
from engine import *
from utils import *



parser = argparse.ArgumentParser(description='Multi_label Training')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='ids of GPU')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--result', default='', type=str, metavar='PATH',
                    help='path to save prediction results (default: none)')



def main():
    global args, best_prec1, device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load data
    train_data_pd, label_names = get_cocoapi('../coco2017', 'train&val', 'train2017')
    val_data_pd, _ = get_cocoapi('../coco2017', 'train&val', 'val2017')
    num_labels = len(label_names)

    train_table = train_data_pd[['label']]
    weight_list = get_label_weight(train_table)

    dataset_size = {'train':train_data_pd.shape[0], 'val': val_data_pd.shape[0]}
    print(dataset_size)
    datasets = {'train':CocoDataset(train_data_pd), 'val':CocoDataset(val_data_pd)}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                  for x in ['train', 'val']}

    model = Resnet_Model(num_labels=num_labels).to(device)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.1)

    # state
    state = {'device': device, 'batch_size': args.batch_size, 'max_epochs': args.epochs,
             'resume': args.resume, 'num_labels': num_labels}
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['device_ids'] = args.device_ids
    state['result_path'] = args.result

    # train model
    engine = Engine(state)
    engine.train_model(model, optimizer_adam, exp_lr_scheduler, dataloaders, dataset_size, weight_list)
    engine.predict2csv(model, dataloaders, label_names, train_table)


if __name__ == '__main__':
    main()
