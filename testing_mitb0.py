#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:53:45 2023

@author: ankita
"""

import segmentation_models_pytorch as smp
import timm
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import shutil
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from PIL import Image
import cv2

torch.manual_seed(0)
torch.cuda.empty_cache()

import sys


use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/home/ankitaC/Ankita/Microsoft/data/publay'

x_val_dir = os.path.join(DATA_DIR, 'val')
y_val_dir = os.path.join(DATA_DIR, 'results_gt_pub')

#x_valid_dir = os.path.join(DATA_DIR, 'val')
#y_valid_dir = os.path.join(DATA_DIR, 'valannot')

class MyDataset(Dataset):    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            transforms = None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        #self.augmentation = augmentation
        #self.preprocessing = preprocessing
        '''self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])'''
    
    def __getitem__(self, i):
        
        # read data
        image = Image.open(self.images_fps[i]).convert('RGB')
        #print(image.size)
        mask = Image.open(self.masks_fps[i])
        
        transformers = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        '''if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']'''
            
        image = train_transforms(image)
        mask = transformers(mask)
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
        


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

       


data_test = MyDataset(x_val_dir, y_val_dir)
val_loader = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False)


model = smp.FPN('mit_b0', encoder_weights="imagenet", encoder_depth=5, classes=1)
model = model.cuda()

from torchmetrics import JaccardIndex
jaccard = JaccardIndex(task="binary", num_classes=1)
jaccard = jaccard.cuda()
def validate(val_loader, net, th):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    iou = AverageMeter('IOU_score', ':6.2f')
    preci = AverageMeter('Precision', ':6.2f')
    progress = ProgressMeter(len(val_loader),[batch_time, losses, iou, preci], prefix='Test: ')
    net.eval()
    with torch.no_grad():
        for i,(images, labels) in enumerate(val_loader):
            images, labels = images.cuda() , labels.cuda()
            outputs = net(images)
            #print(outputs.size())
            output = outputs.round().long()
            output = output.detach().cpu().numpy()
            #cv2.imwrite('test_img/' + str(i) + '.png', output)
            target = labels.round().long()
            img = images.round().long()
            img = img.detach().cpu().numpy()
            #np.save('val_img/' + str(i) + '.npy', img)
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='binary', threshold=th)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            #iou_score = jaccard(outputs, target)
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            #print(iou_score, precision)
            iou.update(iou_score.item(), images.size(0))
            preci.update(precision.item(), images.size(0))
            if i % 5 == 0:
                progress.display(i)
        print(' * IoU {iou.avg:.3f} Precision {preci.avg:.3f}'.format(iou=iou, preci=preci))
    #print(' * IoU: ', iou.avg, 'Precision: ', preci.avg)


model.load_state_dict(torch.load('model_best_mit_b0_pub.pth.tar')['state_dict'])

i = 0.5
while i<=0.95:
    file = open('results1/logfile_pub_mit_b0_' + str(i) + '.txt', 'a')
    sys.stdout = file
    validate(val_loader, model, i)
    file.close()
    i += 0.05

