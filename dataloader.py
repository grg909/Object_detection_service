# -*- coding: UTF-8 -*-

# @Date    : 2019/7/17
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
import numpy as np
import pandas as pd
from PIL import Image
import os
from pprint import pprint
from pycocotools.coco import COCO
import pickle
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#1.set random seed
random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)


class CocoDataset(Dataset):
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
                    T.Resize(513),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomResizedCrop(321),
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(15),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        if self.test:
            filename = self.imgs[index]
            image_path, _ = os.path.splitext(filename)
            image_name = image_path.split('/')[-1]
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.transforms(img)
            return img, image_name
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


def get_cocoapi(root, mode, dataType='val2017', cat='childcategory'):

    #for test
    if mode == "test":
        files = []
        image_root = root + '/images/val2017'
        for img in os.listdir(image_root):
            files.append(image_root + '/' + img)
        files = pd.DataFrame({"filename": files})
        return files
    elif mode != "test":
        #for train and val
        all_image_path, labels = [], []

        print("loading train dataset")
        all_image_id = []
        image_root = root + '/images/{}'.format(dataType)

        # Input annotation from stuffs and things
        annFile_t = '{}/annotations/instances_{}.json'.format(root, dataType)
        annFile_s = '{}/annotations/stuff_{}.json'.format(root, dataType)
        coco_t = COCO(annFile_t)
        coco_s = COCO(annFile_s)

        select = pickle.load(open('{}/select.pkl'.format(root), 'rb'))

        for img in tqdm(os.listdir(image_root)):
            long_id, _ = os.path.splitext(img)
            id = int(long_id[-6:])

            annIds_t = coco_t.getAnnIds(imgIds=id)
            annIds_s = coco_s.getAnnIds(imgIds=id)
            cat_list_t = []
            for i in coco_t.loadAnns(annIds_t):
                cat_list_t.append(i['category_id'])
            cat_set_t = set(cat_list_t)
            key_name = 'name' if cat == 'childcategory' else 'supercategory'
            sc_labels = [i[key_name] for i in coco_t.loadCats(cat_set_t)]
            cat_list_s = []
            for i in coco_s.loadAnns(annIds_s):
                cat_list_s.append(i['category_id'])
            cat_set_s = set(cat_list_s)
            sc_labels.extend([i[key_name] for i in coco_s.loadCats(cat_set_s)])

            if not len(set(select) & set(sc_labels)):
                continue

            id_labels = []
            for i in select:
                if i in sc_labels:
                    id_labels.append(1)
                else:
                    id_labels.append(0)

            all_image_path.append(image_root + '/' + img)
            labels.append(id_labels)

        all_files = pd.DataFrame({"filename": all_image_path, "label": labels})
        return all_files, select
    else:
        print("check the mode please!")

class NewDataset(Dataset):
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
                    T.Resize(513),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomResizedCrop(321),
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(15),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        if self.test:
            filename = self.imgs[index]
            image_path, _ = os.path.splitext(filename)
            image_name = image_path.split('/')[-1]
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.transforms(img)
            return img, image_name
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
            files.append(root + '/' + img)
        files = pd.DataFrame({"filename": files})
        return files
    elif mode != "test":
        #for train and val
        files = []
        label_root = root + '/labels'
        for lb in os.listdir(label_root):
            files.append(label_root + '/' + lb)
        all_image_path, labels = [], []

        select = pickle.load(open('{}/select.pkl'.format(root), 'rb'))
        print("loading train dataset")
        for file in tqdm(files):
            lb_info = pickle.load(open(file, 'rb'))
            all_data_path.append(lb_info[0])
            sc_labels = lb_info[1]
            id_labels = []
            for i in select:
                if i in sc_labels:
                    id_labels.append(1)
                else:
                    id_labels.append(0)
            labels.append(id_labels)

        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")


