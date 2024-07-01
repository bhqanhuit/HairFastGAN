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
from models.face_parsing.model import BiSeNet
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

device = 'cuda'

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def load_bisenet_model():
    n_classes = 19  # Number of classes for segmentation
    model = BiSeNet(n_classes=n_classes)
    model.load_state_dict(torch.load('pretrained_models/BiSeNet/face_parsing_79999_iter.pth'))
    model.eval()
    return model

def visualize_masks(image_path, output_np, overlap_mask):
    image = Image.open(image_path).resize((512, 512))
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.title('Soft Mask')
    plt.imshow(np.argmax(output_np, axis=0), cmap='jet', alpha=0.5)
    
    plt.subplot(1, 3, 3)
    plt.title('Overlap Mask')
    plt.imshow(overlap_mask, cmap='jet', alpha=0.5)
    plt.savefig('foo.png')

    # plt.show()

def apply_crf(image, softmax):
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, softmax.shape[0])
    unary = unary_from_softmax(softmax)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)
    Q = d.inference(5)
    return np.array(Q).reshape((softmax.shape[0], softmax.shape[1], softmax.shape[2]))




def create_overlap_mask(output_np):
    hair_mask = output_np[17, :, :]
    face_mask = output_np[1, :, :]
    overlap_mask = np.minimum(hair_mask, face_mask)
    return overlap_mask


if __name__ == "__main__":
    model = load_bisenet_model()
    print(f'Done loading BiseNet')

    image_path = 'datasets/FFHQ_TrueScale/08173.png'
    input_image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_image)[0]
        output = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        
    output_np = output.cpu().numpy().squeeze()
    overlap_mask = create_overlap_mask(output_np)
    image_np = np.array(Image.open(image_path).resize((512, 512)))
    crf_output = apply_crf(image_np, output_np)
    print(crf_output.shape)
    print(crf_output.dtype)

    visualize_masks(image_path, crf_output, overlap_mask)


    






    