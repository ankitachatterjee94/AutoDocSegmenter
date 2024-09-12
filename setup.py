import segmentation_models_pytorch as smp
import timm
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import models, transforms
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from PIL import Image

torch.manual_seed(0)
torch.cuda.empty_cache()


model = smp.FPN('mit_b0', encoder_weights="imagenet", encoder_depth=5, classes=10)
'''loss = smp.losses.DiceLoss('multiclass')
metrics = [
    smp.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])
'''

transform_train = transforms.Compose([
    transforms.Resize((2048, 1024)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
image = Image.open("00000158.png")
sample = transform_train(image).float().unsqueeze(0)

encoder = model.encoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))
output = model.forward(sample)
#output = np.array(output.detach())
output = (output.detach().squeeze().numpy().round())
print(output.shape)
#np.save('mask.npy', output.detach().numpy())
#print(model.segmentation_head)


