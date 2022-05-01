#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:52:20 2022

@author: sanketbhave
"""

import os, sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from random import randint
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import models


class ImageDataSet(Dataset):
    def __init__(self, train, test, val):
        attributes = pd.read_csv(r"./archive/list_attr_celeba.csv")
        attributes = attributes.replace(-1, 0)
        partition_df = pd.read_csv(r"./archive/list_eval_partition.csv")
        self.dataset = attributes.join(partition_df.set_index('image_id'), on='image_id')
        if train:
            self.dataset = self.dataset.loc[self.dataset['partition'] == 0]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        elif test:
            self.dataset = self.dataset.loc[self.dataset['partition'] == 1]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        elif val:
            self.dataset = self.dataset.loc[self.dataset['partition'] == 2]
            self.images = self.dataset['image_id']
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])
        self.len = len(self.dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = cv2.imread(r"./archive/img_align_celeba/img_align_celeba/" + self.images.iloc[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        atrributes = torch.from_numpy(np.array(self.dataset.iloc[index, 1:41], dtype=np.int32))
        image_id = self.dataset.iloc[index, 0:1].tolist()

        return {
            'image': image,
            'attributes': atrributes,
            'image_id': image_id
        }


"""
give 10 relevant images given a query image
"""


def predict(rank, size):
    desired_image = '162775.jpg'
    n = 10
    output_dict = {}

    check = int(desired_image.split(".")[0]) - 162771
    dataset = ImageDataSet(train=False, test=True, val=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler)

    # train dataloading
    train_dataset = ImageDataSet(train=True, test=False, val=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=train_sampler)

    resnet18 = models.resnet18(pretrained=False)
    num_final_in = resnet18.fc.in_features
    NUM_FEATURES = 40
    resnet18.fc = nn.Sequential(nn.Linear(num_final_in, NUM_FEATURES), nn.Sigmoid())

    model = DDP(resnet18)
    # model = resnet18
    checkpoint = torch.load(r"./model.checkpoint")
    # model.load_state_dict(torch.load(r"./model.checkpoint"))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    for i, data in enumerate(dataloader):
        if i == check:

            # data, target = Variable(data), Variable(target)
            data, target, test_image_id = data['image'], data['attributes'], data['image_id']
            print(len(test_image_id))
            print(test_image_id)
            prediction = model(data)
            print(prediction)
            for pred_val in prediction:
                for train_i, train_data in enumerate(train_dataloader):
                    tr_data, tr_target, tr_image_id = train_data['image'], train_data['attributes'], train_data[
                        'image_id']
                    for train_attr, train_id in zip(tr_target, tr_image_id[0]):

                        cos = torch.nn.CosineSimilarity(dim=0)
                        output = cos(pred_val, train_attr)
                        # print("="*30)
                        if output >= 0.85:
                            print("Cosine Similarity of ", test_image_id[0][0], ",", train_id, " :", output)
                            output_dict[train_id] = [output.item(), train_attr]
                            print(test_image_id[0][0], train_id)

    a = sorted(output_dict.items(), key=lambda x: x[1][0], reverse=True)
    for idx, k in enumerate(a):
        if idx == n: break
        print(k)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'saturn'
    os.environ['MASTER_PORT'] = '29700'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    # torch.manual_seed(40)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    setup(int(sys.argv[1]), int(sys.argv[2]))
    # train(int(sys.argv[1]), int(sys.argv[2]))
    predict(int(sys.argv[1]), int(sys.argv[2]))
