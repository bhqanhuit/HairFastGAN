import torch
import cv2
import numpy as np
from torchvision import transforms
from models.face_parsing.model import BiSeNet
from torchvision.utils import save_image
from PIL import Image

transform_ = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

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

def soft_boundary(mask, boundary_width=13):
    kernel = np.ones((boundary_width, boundary_width), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = dilated - eroded
    soft_mask = cv2.GaussianBlur(boundary.astype(np.float32), (boundary_width, boundary_width), 0)

    return soft_mask

def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    image = preprocess_image(image_path)
    mask = generate_mask(model, image, device)
    hair_mask = extract_hair_region(mask)
    soft_mask = soft_boundary(hair_mask)

    hair_mask = transform_(1 - hair_mask).float()
    hair_mask = torch.where(hair_mask > 0, 1.0, 0.0)
    
    soft_mask = transform_(1 - soft_mask)
    # for i in soft_mask:
    #     for j in i:
    #         for k in j:
    #             print(k)
    
    save_image(soft_mask, 'soft_mask.png', normalize=True)
    save_image(hair_mask, 'hard_mask.png')
    
    print("Soft mask saved as 'soft_mask.png'")

if __name__ == "__main__":
    image_path = "real_input.jpg"
    model_path = "pretrained_models/BiSeNet/face_parsing_79999_iter.pth"

    main(image_path, model_path)
