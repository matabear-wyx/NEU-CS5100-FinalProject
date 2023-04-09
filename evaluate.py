import torch
import torchvision
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import model
import os

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define data directory path
data_dir = "./data/CK+48"

# Define data transforms for data augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=data_transforms)

# Define the data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the saved model
model = model.EmotionDetector().to(device)
model.load_state_dict(torch.load("./output/emotion_detection_model_grayscale_best.pth"))

# Evaluate the model on the test set
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        # Move the input data to the GPU if available
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate the confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=test_dataset.classes)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)