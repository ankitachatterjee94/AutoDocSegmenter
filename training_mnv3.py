import segmentation_models_pytorch as smp
import timm
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torchvision
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

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_anno')

x_val_dir = os.path.join(DATA_DIR, 'val')
y_val_dir = os.path.join(DATA_DIR, 'val_anno')

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
        


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_mv3_pub.pth.tar')

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

def adjust_learning_rate(optimizer, epoch, lrs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrs * (0.001 ** (epoch // 10))
    if lr <= 1e-5:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        


data_train = MyDataset(x_train_dir, y_train_dir)
trainloader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=True)

data_test = MyDataset(x_val_dir, y_val_dir)
val_loader = torch.utils.data.DataLoader(data_test, batch_size=256, shuffle=False)


model = smp.FPN('timm-mobilenetv3_small_100', encoder_weights='imagenet', encoder_depth=5, classes=1)
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes = 1, weights_backbone = 'IMAGENET1K_V1')
model = model.cuda()

criterion = smp.losses.DiceLoss('binary')
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])

criterion = criterion.cuda()


def train(train_loader, net, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    iou = AverageMeter('IOU_score', ':6.2f')
    preci = AverageMeter('Precision', ':6.2f')
    progress = ProgressMeter(len(train_loader),[batch_time, data_time, losses, iou, preci], prefix="Epoch: [{}]".format(epoch))
    net.train()
    for k, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), images.size(0))
        target = labels.round().long()
        tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        iou.update(iou_score.item(), images.size(0))
        preci.update(precision.item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if k % 100 == 0:
            progress.display(k)

def validate(val_loader, net, criterion):
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
            loss = criterion(outputs, labels)
            losses.update(loss.item(), images.size(0))
            target = labels.round().long()
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='binary', threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            iou.update(iou_score.item(), images.size(0))
            preci.update(precision.item(), images.size(0))
            if i % 100 == 0:
                progress.display(i)
        print(' * IoU {iou.avg:.3f} Precision {preci.avg:.3f}'.format(iou=iou, preci=preci))
    return iou.avg, preci.avg, losses.avg

best_acc1 = 0
scheduler = ReduceLROnPlateau(optimizer, 'max')

for epoch in range(30):
    adjust_learning_rate(optimizer, epoch, 0.001)
    train(trainloader, model, criterion, optimizer, epoch)
    acc1, acc5, val_loss = validate(val_loader, model, criterion)
    is_best = acc5 > best_acc1
    best_acc1 = max(acc5, best_acc1)
    scheduler.step(best_acc1)
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'weights/mnv3',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)




