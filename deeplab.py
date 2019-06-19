# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""

"""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from addict import Dict
import os
from tqdm import tqdm

from libs.models import *
from libs.utils import DenseCRF


class DeeplabPytorch:

    def __init__(self, config_path, model_path):

        self.config_path = config_path
        self.model_path = model_path

    @staticmethod
    def get_device(cuda):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device))
        else:
            print("Device: CPU")
        return device

    @staticmethod
    def get_classtable(CONFIG):
        with open(CONFIG.DATASET.LABELS) as f:
            classes = {}
            for label in f:
                label = label.rstrip().split("\t")
                classes[int(label[0])] = label[1].split(",")[0]
        return classes

    @staticmethod
    def setup_postprocessor(CONFIG):
        # CRF post-processor
        postprocessor = DenseCRF(
            iter_max=CONFIG.CRF.ITER_MAX,
            pos_xy_std=CONFIG.CRF.POS_XY_STD,
            pos_w=CONFIG.CRF.POS_W,
            bi_xy_std=CONFIG.CRF.BI_XY_STD,
            bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
            bi_w=CONFIG.CRF.BI_W,
        )
        return postprocessor

    @staticmethod
    def preprocessing(image, device, CONFIG):
        # Resize
        scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        raw_image = image.astype(np.uint8)

        # Subtract mean values
        image = image.astype(np.float32)
        image -= np.array(
            [
                float(CONFIG.IMAGE.MEAN.B),
                float(CONFIG.IMAGE.MEAN.G),
                float(CONFIG.IMAGE.MEAN.R),
            ]
        )

        # Convert to torch.Tensor and add "batch" axis
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.to(device)

        return image, raw_image

    @staticmethod
    def inference(model, image, raw_image=None, postprocessor=None):
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs)

        labelmap = np.argmax(probs, axis=0)

        return labelmap

    def single(self, image_path, cuda=True, crf=False):

        # Setup
        CONFIG = Dict(yaml.load(open(self.config_path, 'r', encoding='UTF-8'), Loader=yaml.FullLoader))
        device = self.get_device(cuda)
        torch.set_grad_enabled(False)
        postprocessor = self.setup_postprocessor(CONFIG) if crf else None

        model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print("Model:", CONFIG.MODEL.NAME)

        # Inference
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image, raw_image = self.preprocessing(image, device, CONFIG)
        labelmap = self.inference(model, image, raw_image, postprocessor)
        print('labelmap', labelmap.shape)

        return labelmap

    def gen_labelmap(self, path, cuda=True, crf=False):
        """
        Inference from a single image
        """

        # Setup
        CONFIG = Dict(yaml.load(open(self.config_path, 'r', encoding='UTF-8')))
        device = self.get_device(cuda)
        torch.set_grad_enabled(False)
        postprocessor = self.setup_postprocessor(CONFIG) if crf else None

        model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print("Model:", CONFIG.MODEL.NAME)

        for dir, dirname, filenames in os.walk(path, topdown=True):
            for filename in tqdm(filenames):
                image = cv2.imread(path+'/'+filename, cv2.IMREAD_COLOR)
                image, raw_image = self.preprocessing(image, device, CONFIG)
                labelmap = self.inference(model, image, raw_image, postprocessor)

                yield filename, labelmap


if __name__ == '__main__':

    dp = DeeplabPytorch(config_path='configs/cocostuff164k.yaml', model_path='data/models/coco/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')
    dp.single('IMG_2885.JPG')
