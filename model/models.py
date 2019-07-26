#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:48:27 2019

@author: xxktt
"""

import torch
import torchvision
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
def Resnet_Model(num_labels):
    model = torchvision.models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_in_features = model.fc.in_features
    model.fc = nn.Linear(num_in_features, num_labels*2)

    model = model.to(device)

    return model

