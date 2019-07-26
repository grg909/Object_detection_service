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
        for img in os.listdir(image_root):
            all_image_path.append(image_root + '/' + img)
            long_id, _ = os.path.splitext(img)
            all_image_id.append(int(long_id[-6:]))

        annFile = '{}/annotations/instances_{}.json'.format(root, dataType)
        coco = COCO(annFile)
        label_names = pickle.load(open('coco2017/{}.pkl'.format(cat), 'rb'))

        for id in tqdm(all_image_id):
            annIds = coco.getAnnIds(imgIds=id)
            cat_list = []
            for i in coco.loadAnns(annIds):
                cat_list.append(i['category_id'])
            cat_set = set(cat_list)
            key_name = 'name' if cat == 'childcategory' else 'supercategory'
            sc_labels = [i[key_name] for i in coco.loadCats(cat_set)]
            id_labels = []
            for i in label_names:
                if i in sc_labels:
                    id_labels.append(1)
                else:
                    id_labels.append(0)
            labels.append(id_labels)

        all_files = pd.DataFrame({"filename": all_image_path, "label": labels})
        return all_files, label_names
    else:
        print("check the mode please!")


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


