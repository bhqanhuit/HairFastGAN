import cv2
import dlib
import numpy as np
from torchvision.utils import save_image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')  # Download this file from dlib's model repository

image_path = 'datasets/FFHQ_TrueScale/08173.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

faces = detector(image_rgb)

def shape_to_np(shape):
    coords = np.zeros((81, 2), dtype=int)
    for i in range(81):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

landmarks = []
for face in faces:
    shape = predictor(image_rgb, face)
    landmarks.append(shape_to_np(shape))
    
    for num in range(shape.num_parts):
        print(landmarks[0][num][0], landmarks[0][num][1])
        cv2.circle(image_rgb, (landmarks[0][num][0], landmarks[0][num][1]), 3, (0,255,0), -1)


forehead_region = []
num = 0
forehead_region.append([landmarks[0][num][0], landmarks[0][num][1]])
for num in range(17, 27):
    forehead_region.append([landmarks[0][num][0], landmarks[0][num][1]])

forehead_region.append([landmarks[0][78][0], landmarks[0][78][1]])
forehead_region.append([landmarks[0][74][0], landmarks[0][74][1]])
forehead_region.append([landmarks[0][79][0], landmarks[0][79][1]])
forehead_region.append([landmarks[0][73][0], landmarks[0][73][1]])
forehead_region.append([landmarks[0][72][0], landmarks[0][72][1]])
forehead_region.append([landmarks[0][80][0], landmarks[0][80][1]])
forehead_region.append([landmarks[0][71][0], landmarks[0][71][1]])
forehead_region.append([landmarks[0][70][0], landmarks[0][70][1]])
forehead_region.append([landmarks[0][69][0], landmarks[0][69][1]])
forehead_region.append([landmarks[0][68][0], landmarks[0][68][1]])
forehead_region.append([landmarks[0][76][0], landmarks[0][76][1]])
forehead_region.append([landmarks[0][75][0], landmarks[0][75][1]])
forehead_region.append([landmarks[0][77][0], landmarks[0][77][1]])

print(forehead_region)

forehead_region = np.array(forehead_region)

cv2.fillConvexPoly(image_rgb, forehead_region, 255)
cv2.imwrite('hahahaha.png', image_rgb)