import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models.face_parsing.model import BiSeNet
from torchvision.utils import save_image

# Load the pre-trained BiSeNet model
model = BiSeNet(n_classes=19)  # 19 is the number of classes in the pre-trained model
model.load_state_dict(torch.load('pretrained_models/BiSeNet/face_parsing_79999_iter.pth'))
model.eval()

# Function to preprocess the image
def preprocess(image):
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess_transform(image).unsqueeze(0)

# Function to post-process the segmentation output
def postprocess(output, threshold=0.5):
    output = output.squeeze(0).cpu().numpy()
    face_mask = (output == 1)  # Assuming class 1 is the face
    hair_mask = (output == 2)  # Assuming class 2 is the hair

    combined_mask = face_mask.astype(np.float32) + hair_mask.astype(np.float32)
    combined_mask = np.clip(combined_mask, 0, 1)

    # soft_mask = cv2.GaussianBlur(combined_mask, (15, 15), 1)
    return combined_mask

# Read the input image
image_path = 'real_image.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocess the image
input_image = preprocess(image)

# Perform inference
with torch.no_grad():
    output = model(input_image)[0]

# Post-process the output to get the soft mask
soft_mask = postprocess(output)

# Convert soft mask to 3 channels and save it
soft_mask = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2) * 255
soft_mask = soft_mask.astype(np.uint8)
cv2.imwrite('soft_mask.png', soft_mask)

# Optionally, visualize the result
# cv2.imshow('Soft Mask', soft_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
