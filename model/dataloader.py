# -*- coding: UTF-8 -*-

# @Date    : 2019/7/17
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.7
"""

"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
import numpy as np
import pandas as pd
from PIL import Image
import os
from pprint import pprint


#1.set random seed
random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)


class PosterDataset(Dataset):
    def __init__(self, image_list, transforms=None, train=True, test=False):
        super().__init__()
        self.test = test
        self.train = train
        imgs = []
        if self.test:
            for index, row in image_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs
        else:
            for index, row in image_list.iterrows():
                imgs.append((row["filename"], row["label"]))
            self.imgs = imgs
        if transforms is None:
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            else:
                self.transforms = T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    # T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        if self.test:
            filename = self.imgs[index]
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, filename
        else:
            filename, label = self.imgs[index]
            image_path, _ = os.path.splitext(filename)
            image_name = image_path.split('/')[-1]
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.transforms(img)
            label = np.array(label).astype(np.int64)
            return img, image_name, label

    def __len__(self):
        return len(self.imgs)


def get_files(root, mode):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename": files})
        return files
    elif mode != "test":
        #for train and val
        all_image_path, labels = [], []
        # image_folders = list(map(lambda x:root+x,os.listdir(root)))
        # all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))

        print("loading train dataset")
        anno = pd.read_csv(root)
        label_names = anno.columns.tolist()[2:]
        for row in anno.itertuples(index=False):
            image_path = 'data/Images/' + row[0]+'.jpg'
            all_image_path.append(image_path)
            labels.append(row[2:])
        all_files = pd.DataFrame({"filename": all_image_path, "label": labels})
        return all_files, label_names
    else:
        print("check the mode please!")


