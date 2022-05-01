#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:36:37 2022

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

"""
Custom Dataset class
"""


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

        return {
            'image': image,
            'attributes': atrributes
        }


"""
input: rank and size
output: dataloader
partitions the dataset into 'size' different partitions and returns the one with appropriate rank
"""


def get_dataloader(rank, size):
    dataset = ImageDataSet(train=True, test=False, val=False)
    sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)
    return dataloader


"""
average the gradients over all the nodes
"""


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


"""
input: rank and size
output: none
train our model on the assigned data and store it as a checkpoint
"""


def train(rank, world_size):
    torch.manual_seed(1234)
    dataloader = get_dataloader(rank, world_size)
    resnet18 = models.resnet18(pretrained=False)
    for param in resnet18.parameters():
        param.requires_grad = True
    num_final_in = resnet18.fc.in_features
    NUM_FEATURES = 40
    resnet18.fc = nn.Sequential(nn.Linear(num_final_in, NUM_FEATURES), nn.Sigmoid())
    model = DDP(resnet18, find_unused_parameters=True).float().cuda()
    # if(torch.cuda.is_available()):
    #     model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # def loss_function(feat1, feat2):
    #     # minimize average cosine similarity
    #     return F.cosine_similarity(feat1, feat2).mean()
    # loss_function = nn.CosineSimilarity(dim=-1)
    loss_function = nn.BCELoss()
    print("This node has data of size: ", len(dataloader))
    for epoch in range(10):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader):
            model.train()
            # data, target = Variable(data), Variable(target)
            data, target = data['image'].cuda(), data['attributes'].cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target.float())
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            print('Rank ',
                  dist.get_rank(), ', step', i, ', epoch ', epoch, ': ',
                  loss.item())
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss/len(dataloader))

    if rank == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, r"./Big Data/model.checkpoint")


"""
setup the node, tell it it's master and rank
"""


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'earth'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    setup(int(sys.argv[1]), int(sys.argv[2]))
    train(int(sys.argv[1]), int(sys.argv[2]))
