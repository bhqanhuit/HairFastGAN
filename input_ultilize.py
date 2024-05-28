from typing import Set, List

import os
import random
import shutil
import argparse

import torch
import numpy as np

from torchvision.utils import save_image
from torch.utils.data import dataloader
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torchvision

device = 'cuda'

image_transform = transforms.Compose([
                            # transforms.Resize((1024, 1024)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

if __name__ == "__main__":
    source_path = 'IMG_4129_Large.jpeg'

    im = Image.open(source_path)
    print(im)

    width, height = im.size   # Get dimensions

    left = (width - 850)/2
    top = (height - 800)/2
    right = (width + 850)/2
    bottom = (height + 900)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im = image_transform(im)
    im = F.interpolate(im.unsqueeze(0), size=(1024, 1024), mode='bicubic')
    save_image(im, 'test_img.png', normalize=True)
    print(im.shape)
    
    






    