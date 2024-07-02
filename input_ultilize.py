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
import cv2

device = 'cuda'

transform_ = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

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

# def visualize_results(image, hair_mask, face_mask, blended_image):
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 4, 1)
#     plt.title('Original Image')
#     plt.imshow(image)
#     plt.axis('off')
    
#     plt.subplot(1, 4, 2)
#     plt.title('Hair Mask')
#     plt.imshow(hair_mask, cmap='gray')
#     plt.axis('off')
    
#     plt.subplot(1, 4, 3)
#     plt.title('Face Mask')
#     plt.imshow(face_mask, cmap='gray')
#     plt.axis('off')
    
#     plt.subplot(1, 4, 4)
#     plt.title('Blended Image')
#     plt.imshow((blended_image * 255).astype(np.uint8))
#     plt.axis('off')
    
#     plt.savefig('foo.png')




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

def create_soft_mask(mask, kernel_size=21):
    soft_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return soft_mask

def blend_regions(image, hair_mask, face_mask, soft_hair_mask, soft_face_mask):
    # Ensure the masks are 3-channel
    hair_mask_3ch = np.stack([hair_mask]*3, axis=-1)
    face_mask_3ch = np.stack([face_mask]*3, axis=-1)
    soft_hair_mask_3ch = np.stack([soft_hair_mask]*3, axis=-1)
    soft_face_mask_3ch = np.stack([soft_face_mask]*3, axis=-1)

    # Blend the regions
    blended_image = (soft_hair_mask_3ch * hair_mask_3ch * image + 
                     soft_face_mask_3ch * face_mask_3ch * image) / (soft_hair_mask_3ch + soft_face_mask_3ch + 1e-8)
    return blended_image


def detect_hairline(face_mask, hair_mask):
    # Ensure the masks are binary
    face_mask_binary = (face_mask > 0.5).astype(np.uint8)
    hair_mask_binary = (hair_mask > 0.5).astype(np.uint8)

    # save_image(torch.cat([transform_(hair_mask_binary), transform_(hair_mask_binary), transform_(hair_mask_binary)], dim=0), 'hahahaa.jpg', normalize=True)
    # Find contours in the face mask
    contours, _ = cv2.findContours(face_mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    
    
    face_contour = max(contours, key=cv2.contourArea)
    print(face_contour)
    # Create a mask for the face contour
    face_contour_mask = np.zeros_like(face_mask_binary)
    cv2.drawContours(face_contour_mask, [face_contour], -1, 1, thickness=cv2.FILLED)
    

    # Find the intersection of the hair and face masks
    intersection_mask = face_contour_mask * hair_mask_binary
    save_image(torch.cat([transform_(intersection_mask), transform_(intersection_mask), transform_(intersection_mask)], dim=0), 'hahahahah.png', normalize=True)
    print(intersection_mask.shape)

    # Find the hairline by detecting the upper boundary of the intersection
    hairline_points = np.where(intersection_mask > 0)
    if len(hairline_points[0]) == 0:
        return None, None

    hairline_y = np.min(hairline_points[0])
    hairline_x = np.unique(hairline_points[1])

    return hairline_x, hairline_y




if __name__ == "__main__":
    model = load_bisenet_model()
    print(f'Done loading BiseNet')

    image_path = 'datasets/FFHQ_TrueScale/08173.png'
    input_image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_image)[0]
        output = F.softmax(output, dim=1)  
    
    output_np = output.cpu().numpy().squeeze()
    hair_mask = output_np[17, :, :]
    face_mask = output_np[1, :, :]

    # Resize masks to the original image size
    hair_mask = cv2.resize(hair_mask, (512, 512))
    face_mask = cv2.resize(face_mask, (512, 512))


    output_np = output.cpu().numpy().squeeze()
    overlap_mask = create_overlap_mask(output_np)
    image_np = np.array(Image.open(image_path).resize((512, 512)))
    crf_output = apply_crf(image_np, output_np)
    
    # visualize_masks(image_path, output_np, overlap_mask)

    hairline_x, hairline_y = detect_hairline(face_mask, hair_mask)


    if hairline_x is not None and hairline_y is not None:
        original_image = Image.open(image_path).resize((512, 512))
        original_image_np = np.array(original_image)
        for x in hairline_x:
            cv2.circle(original_image_np, (x, hairline_y), 1, (255, 0, 0), -1)

        plt.imshow(original_image_np)
        plt.axis('off')
        plt.title('Detected Hairline')
        plt.savefig('foo.png')
    else:
        print("Hairline could not be detected.")


    






    