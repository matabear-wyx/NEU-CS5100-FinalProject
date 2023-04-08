import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import model

# Define input image path and trained model file path
image_path = "./data/test/"
model_path = "./output/emotion_detection_model_grayscale.pth"

# Define data transforms for input images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define class labels
classes = ["Angry", "Fear", "Happy", "Sadness", "Surprise"]

# Load the trained model and move it to the CPU
if not os.path.isfile(model_path):
    print(f"Error: Trained model file {model_path} does not exist")
    exit()

state_dict = torch.load(model_path, map_location=torch.device('cpu'))

model = model.EmotionDetector()
model.load_state_dict(state_dict)
model.to("cpu")
model.eval()

# Load the input images and classify them
for filename in os.listdir(image_path):
    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(image_path, filename)
    img = Image.open(img_path)

    with torch.no_grad():
        img_tensor = data_transforms(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        confidence = np.exp(outputs[0][predicted]) / np.exp(outputs[0]).sum()

    # Show the predicted class and confidence level
    print(f"Image: {filename}, Predicted class: {classes[predicted]}, Confidence level: {confidence.item():.2f}")

