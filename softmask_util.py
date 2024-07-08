import torch
import cv2
import numpy as np
from torchvision import transforms
from models.face_parsing.model import BiSeNet
from torchvision.utils import save_image
from PIL import Image
import torch.nn as nn

transform_ = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((64, 64))
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

transform_down64 = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((64, 64))
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

transform_down512 = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((512, 512))
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])



def transform_up512(image):
    output = nn.functional.interpolate(image.unsqueeze(0), scale_factor=8, mode='bilinear')
    return output

def load_model(model_path, device):
    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

def generate_mask(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)[0]
        output = output.squeeze(0).cpu().numpy()
        mask = output.argmax(0)
    return mask

def extract_hair_region(mask):
    hair_class_index = 17  # Assuming the hair class index is 17
    hair_mask = (mask == hair_class_index).astype(np.uint8)
    return hair_mask


def soft_boundary(mask, boundary_width=35):
    kernel = np.ones((boundary_width, boundary_width), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = dilated - eroded
    # boundary = torch.where(boundary > 0, 1, 0)
    soft_mask = cv2.GaussianBlur(boundary.astype(np.float32), (boundary_width, boundary_width), 0)

    return boundary



def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    image = preprocess_image(image_path)
    mask = generate_mask(model, image, device)
    hair_mask = extract_hair_region(mask)
    soft_mask = soft_boundary(hair_mask)

    hair_mask = transform_(hair_mask)
    hair_mask = torch.where(hair_mask > 0, 1.0, 0.0)
    hair_mask_512 = transform_down512(hair_mask).float()
    hair_mask = transform_down64(hair_mask).float()
    # print(hair_mask_512.shape)
    
    soft_mask = transform_(soft_mask)
    soft_mask = torch.where(soft_mask > 0, 1.0, 0.0)
    soft_mask_512 = transform_down512(soft_mask).float()
    soft_mask = transform_down64(soft_mask).float()
    

    hard_mask = torch.where((hair_mask - soft_mask) == 1.0, 1.0, 0.0)
    hard_mask_512 = torch.where((hair_mask_512 - soft_mask_512) == 1.0, 1.0, 0.0)
    # print(soft_mask.shape)

    
    blank_image = np.zeros((64, 64), dtype=np.float32)
    for x in range(64):
        for y in range(64):
            distance = 1000000000.0
            x_true = 0
            y_true = 0
            if (soft_mask[:, x, y]): 
                for x_hard in range(64):
                    for y_hard in range(64):
                        if (hard_mask[:, x_hard, y_hard]):
                            cal_distance = np.sqrt((x - x_hard) ** 2 + (y - y_hard) ** 2)
                            if (cal_distance < distance):
                                distance = cal_distance
                                x_true = x_hard
                                y_true = y_hard
                

                # print(distance)
                blank_image[x, y] = max(10 - distance, 3)               
                # exit()


    soft_mask2 = cv2.normalize(blank_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    soft_mask = transform_(soft_mask2)
    soft_mask = transform_up512(soft_mask)
    cv2.imwrite('soft_mask_circle.png', (soft_mask2 * 255).astype(np.uint8))

    soft_mask = torch.where(soft_mask == 0, 1, 1 - soft_mask)
    # soft_mask = torch.where(hard_mask_512 == 1, 0, soft_mask)

    save_image(soft_mask, 'soft_mask.png', normalize=True)
    save_image(hair_mask_512, 'hair_mask.png')
    save_image(hard_mask_512, 'hard_mask.png')
    
    print("Soft mask saved as 'soft_mask.png'")

if __name__ == "__main__":
    image_path = "real_image.jpg"
    model_path = "pretrained_models/BiSeNet/face_parsing_79999_iter.pth"

    # image = Image.open('test_outputsFull/0000000070.jpg')
    # image = transform_(image)
    # save_image(image[:, :, 2048:], 'real_image.jpg')

    main(image_path, model_path)
