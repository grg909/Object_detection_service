# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""
    web service predict model
"""

from __future__ import absolute_import, division, print_function

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from dataloader import *
from models import *
from utils import *


class XxkttApi:

    def __init__(self, checkpoint_path, label_path, cuda=True):

        # setup model
        device = self.__get_device(cuda)
        torch.set_grad_enabled(False)
        try:
            self.label_names = pickle.load(open(label_path, 'rb'))
        except:
            print("Please load the label pkl file")
            raise

        self.num_labels = len(self.label_names)
        model = Resnet_Model(num_labels=self.num_labels)
        try:
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            model.to(device)
            self.model = model
            print("Model:{}".format(checkpoint_path))
        except:
            print("Please load the checkpoint of model")
            raise

        self.device = device

    @staticmethod
    def __get_device(cuda):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device))
        else:
            print("Device: CPU")
        return device

    def single(self, file_path):

        origin_files = pd.DataFrame({"filename": [file_path] })
        image_data = DataLoader(CocoDataset(origin_files, train=False, test=True), batch_size=1)
        for inputs, image_name in image_data:
            inputs = inputs.to(device)
            outputs = self.model(inputs)
            probs = torch.softmax(outputs.view(outputs.size(0) * self.num_labels, 2), dim=1)
            _, preds = torch.max(probs, 1)
            preds = preds.cpu().numpy()
            pred_tags = Pred2tag(preds, inputs, self.num_labels, self.label_names)

        return pred_tags


if __name__ == '__main__':

    dp = XxkttApi(checkpoint_path='checkpoints/checkpint_0729.tar', label_path='coco2017/select.pkl')
    dp.single('static')
