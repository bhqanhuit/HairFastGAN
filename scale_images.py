import cv2
import dlib
import numpy as np
from torchvision.utils import save_image
import torch
import torchvision.transforms as transforms

def get_facial_landmarks(image):
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's website

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected")
        return None

    # Assume only one face in the image
    face = faces[0]
    
    # Determine facial landmarks
    shape = predictor(gray, face)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    return landmarks

def scale_face_with_landmarks(img1, img2):
    # Get facial landmarks for both images
    landmarks1 = get_facial_landmarks(img1)
    landmarks2 = get_facial_landmarks(img2)

    if landmarks1 is None or landmarks2 is None:
        return None

    # Calculate the mean distance between corresponding landmarks
    distances1 = np.linalg.norm(landmarks1[36:42] - landmarks1[42:48], axis=1)
    distances2 = np.linalg.norm(landmarks2[36:42] - landmarks2[42:48], axis=1)
    mean_distance1 = np.mean(distances1)
    mean_distance2 = np.mean(distances2)

    # Calculate scaling factor
    scale_factor = mean_distance2 / mean_distance1
    print(scale_factor)

    # Scale image 1
    resized_img1 = cv2.resize(img1, None, fx=scale_factor, fy=scale_factor)

    return resized_img1

def AddPadding(img):
    # img = cv2.resize(img, (0,0), fx=0.7, fy=0.7) 
    old_image_height, old_image_width, channels = img.shape 
    
    new_image_width = 1024
    new_image_height = 1024
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return img
    color = (255,255,255)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2   

    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width, :] = img

    return result

def center_crop(img, dim):
	width, height = img.shape[1], img.shape[0]

	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
     
	return crop_img


def image_scale(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
        
    resized_face = scale_face_with_landmarks(img2, img1)

    # resized_face = AddPadding(resized_face)
    resized_face = center_crop(resized_face, (1024, 1024))
    # print(resized_face.shape)
    image_transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)


    img1 = image_transform(img1)
    img2 = image_transform(img2)
    resized_face = image_transform(resized_face)
    # print(resized_face.shape)

    res_con = torch.cat([img1, img2, resized_face], dim=2)
    save_image(res_con, 'temp/haha.jpg', normalize=True)

    return resized_face




