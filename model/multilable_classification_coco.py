#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:35:26 2019

@author: jiaruizheng, WANG JINGE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import time
import copy
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
from pprint import pprint
from dataloader import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Visualize some images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title('true: {}'.format(title))
        plt.pause(0.001)  # pause a bit so that plots are updated


# calculate loss
def Loss(probs, labels, sample_number_0, sample_number_1):

    loss = 0.0
    N = dataset_size['train']
    for i in range(labels.size()[1]):
        weight_i = torch.FloatTensor([1.0*N/sample_number_0[i], 1.0*N/sample_number_1[i]]).to(device)
        probs = probs.view(labels.size()[0], -1)
        #criterion = nn.CrossEntropyLoss(weight=weight_i)
        loss += F.cross_entropy(probs[:, i*2:(i+1)*2], labels[:, i], weight=weight_i)
        #print('loss:',loss)

    return loss


# Model
def MLModel(num_labels):
    resnet_model = torchvision.models.resnet101(pretrained=True)
    for param in resnet_model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_in_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_in_features, num_labels*2)

    resnet_model = resnet_model.to(device)

    return resnet_model


# Train and Validation
def train_model(model, optimizer, scheduler, num_epochs=25, loss=None):
    since = time.time()

    loss = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, img_name, labels in dataloaders[phase]:
                labels = torch.LongTensor(labels)
                #print(img_name)
                #print(labels)
                #print(type(labels))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print('outputs',outputs)
                    #print(type(outputs))
                    probs = torch.softmax(outputs.view(outputs.size(0)*num_labels,2), dim=1)
                    #print('probs:',probs)
                    #preds = probs[:,1]>t
                    _, preds = torch.max(probs, 1)
                    preds = preds.to(device)
                    #print('preds',preds)
                    #print('labels:',labels.view(-1))
                    loss = Loss(probs,labels,sample_number_0, sample_number_1)/num_labels
                    #print('loss:',loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                #print('preds',preds.dtype)
                #print('labels',labels.dtype)
                running_corrects += torch.sum(preds == labels.view(-1))
                    # add all the correctly predictded labels in one batch

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / (dataset_size[phase] * num_labels)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                loss = epoch_loss

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Loss: {:4f}'.format(loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Help function to generate tags from predicion outputs
def pred2tag(arr, inputs):
    pred_tags = []
    for i in range(inputs.size()[0]):
        arr_i = arr[i*num_labels:i*num_labels+num_labels]
        index = np.where(arr_i == 1)
        index = index[0].tolist()
        pred_tag = [label_names[j] for j in index]
        pred_tags.append(pred_tag)
    return pred_tags


# Predict and format result in csv
def predict2csv(model, num_images=None, t=0.5):
    was_training = model.training
    model.eval()
    images_so_far = 0
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
            pred_tags = pred2tag(preds, inputs)

            for j in range(inputs.size()[0]):

                tag = [label_names[i] for i, label in enumerate(labels.cpu().numpy()[j]) if int(label) == 1]
                data = {'Id': img_name[j], 'Tag': tag, 'Predict_tag':','.join(pred_tags[j])}
                for x in range(num_labels):
                    data[label_names[x]] = preds[j*num_labels+x]
                result_df.loc[result_df.shape[0] + 1] = data
                images_so_far += 1

                # if pred_tags[j] != []:
                #     plt.title(str(i) + str(j) + 'true: {}'.format(tag[j]) + '  pred: {}'.format(pred_tags[j]))
                # else:
                #     plt.title(str(i) + str(j) + 'true: {}'.format(tag[j]) + '  pred: None')
                # out = torchvision.utils.make_grid(inputs[j])
                # imshow(out)
                # plt.show()

        result_df.to_csv('val统计coco5K_full_para.csv', encoding='utf_8_sig')
        model.train(mode=was_training)
        return


def get_sample_list(train_pd):

    label_list = []
    for i in train_pd.itertuples(index=False):
        label_list.append(i[0])
    train = pd.DataFrame(label_list)
    sample_number_0 = []
    sample_number_1 = []
    for i in train.columns:
        num_0 = len(train[train[i] == 0])
        num_1 = len(train[train[i] == 1])
        sample_number_0.append(num_0)
        sample_number_1.append(num_1)
    return sample_number_0, sample_number_1

########################################################################


if __name__ == '__main__':

    # Load data
    origin_files, label_names = get_cocoapi('coco2017', 'train&val')
    num_labels = len(label_names)

    train_data_pd = origin_files.sample(frac=0.9, random_state=2019)
    val_data_pd = origin_files.drop(train_data_pd.index)

    sample_number_0, sample_number_1 = get_sample_list(train_data_pd[['label']])
    dataset_size = {'train':train_data_pd.shape[0], 'val': val_data_pd.shape[0]}

    datasets = {'train':PosterDataset(train_data_pd), 'val':PosterDataset(val_data_pd)}
    dataloaders = {x: DataLoader(datasets[x], batch_size=4, shuffle=True)
                  for x in ['train', 'val']}

    # Visualize a batch of training dat a
    # images, image_name, labels = next(iter(dataloaders['train']))
    # print(image_name, labels)
    # for i in range(images.size()[0]):
    #     out = torchvision.utils.make_grid(images[i])
    #     imshow(out, labels[i])

    # Train model
    resnet_model = MLModel(num_labels = num_labels)
    #criterion = nn.BCELoss(weight=loss_weight)
    optimizer = optim.SGD(resnet_model.fc.parameters(), lr=0.001, momentum=0.9)
    optimizer_adam = optim.Adam(resnet_model.fc.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_adam, step_size=5, gamma=0.1)
    ml_model = train_model(resnet_model, optimizer_adam, exp_lr_scheduler, num_epochs=15)

    # predict and visualize
    predict2csv(ml_model)


