#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:02:18 2019

@author: xxktt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import random
import time
import copy
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from utils import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Engine(object):
    def __init__(self, state={}):
        self.state = state

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 20

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        if self._state('loss') is None:
            self.state['loss'] = 0.0

        if self._state('best_f1_measure') is None:
            self.state['best_f1_measure'] = 0.0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def load_checkpoint(self,model,checkpoint_path):
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.state['start_epoch'] = checkpoint['epoch']
            self.state['best_f1_measure'] = checkpoint['best_f1_measure']
            self.state['loss'] = checkpoint['loss']
            state_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, self._state('start_epoch')))
            print("best_f1_measure '{}'".format(self._state('best_f1_measure')))

    def save_checkpint(self, state_dict, num_epochs, best_f1_measure, loss, checkpoint_path):
        torch.save({
                'epoch': num_epochs,
                'state_dict': state_dict,
                'best_f1_measure': best_f1_measure,
                'loss': loss,
            }, checkpoint_path )
        print('=> checkpoint saved')

    def train_model(self,model,optimizer,lr_scheduler,dataloaders,dataset_size,weight_list):
        device = self._state('device')
        print('device:', device)
        # load checkpoint
        checkpoint_path = self._state('resume')
        num_labels = self._state('num_labels')

        model = model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print('use multi GPUs with {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        else:
            print('single GPU')

        if checkpoint_path != None:
            self.load_checkpoint(model,checkpoint_path)

        since = time.time()

        loss = self._state('loss')
        best_f1_measure = self._state('best_f1_measure')
        best_model_wts = copy.deepcopy(model.state_dict())

        start_epoch = self._state('start_epoch')
        num_epochs = self._state('max_epochs')

        for epoch in range(start_epoch, num_epochs):
            lr_scheduler.step()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                TP = 0
                FP = 0
                FN = 0

                # Iterate over data.
                for inputs, img_name, labels in tqdm(dataloaders[phase]):
                    labels = torch.LongTensor(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        probs = torch.softmax(outputs.view(outputs.size(0)*num_labels,2), dim=1)
                        _, preds = torch.max(probs, 1)
                        preds = preds.to(device)
                        loss = Loss(probs,labels,weight_list)/num_labels

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.view(-1))

                    # add all the correctly predictded labels in one batch
                    tp, fp, fn = Evaluate(preds, labels)
                    TP += tp
                    FP += fp
                    FN += fn

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / (dataset_size[phase] * num_labels)
                epoch_precision = TP/(TP+FP)
                epoch_recall = TP/(TP+FN)
                epoch_f1_measure = 2*TP/(2*TP+FP+FN)

                print('{} Loss: {:.4f} Acc: {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                print('F1: {:.4F} Precision: {:.4f} Recall: {:.4f} '.format(
                    epoch_f1_measure, epoch_precision, epoch_recall))

                # deep copy the model
                if phase == 'val' and epoch_f1_measure > best_f1_measure:
                    best_f1_measure = epoch_f1_measure
                    best_model_wts = copy.deepcopy(model.state_dict())
                    loss = epoch_loss


        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best F1 measure: {:4f}'.format(best_f1_measure))
        print('Loss: {:4f}'.format(loss))

        state_dict = model.state_dict()
        # save checkpoint
        if checkpoint_path:
            self.save_checkpint(state_dict, num_epochs, best_f1_measure, loss, checkpoint_path)

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model


    # Predict and format result in csv
    def predict2csv(self, model, dataloaders, label_names, train_table):
        device = self._state('device')
        num_labels = self._state('num_labels')
        result_path = self._state('result_path')
        model.eval()
        columns = ['Id', 'Tag', 'Predict_tag']
        columns.extend(label_names)
        result_df = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i, (inputs, img_name, labels) in enumerate(dataloaders['val']):

                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs.view(outputs.size(0)*num_labels, 2), dim=1)
                _, preds = torch.max(probs, 1)
                preds = preds.cpu().numpy()
                pred_tags = Pred2tag(preds, inputs,num_labels,label_names)

                for j in range(inputs.size()[0]):

                    tag = [label_names[i] for i, label in enumerate(labels.cpu().numpy()[j]) if int(label) == 1]
                    data = {'Id': img_name[j], 'Tag': tag, 'Predict_tag':','.join(pred_tags[j])}
                    for x in range(num_labels):
                        data[label_names[x]] = preds[j*num_labels+x]
                    result_df.loc[result_df.shape[0] + 1] = data

        result_df.to_csv(result_path, encoding='utf_8_sig')
        return result_df
