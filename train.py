import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import model

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define data directory path
data_dir = "./data/CK+48"

# Define data transforms for data augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the dataset
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

# Create the model and move it to the GPU if available
model = model.EmotionDetector().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
train_losses, val_losses, val_accuracy_list = [], [], []
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        # Move the input data to the GPU if available
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    model.eval()
    correct_predictions, total_predictions = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
            # Move the input data to the GPU if available
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracy = correct_predictions / total_predictions
    val_accuracy_list.append(val_accuracy)
    if val_accuracy >= 0.95:  # If validation accuracy is 95% or higher, stop training and save the model
        print(f'Validation accuracy of {val_accuracy:.4f} reached. Stopping training.')
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(model.state_dict(), "./output/emotion_detection_model_grayscale_best.pth")
        break  # Exit the training loop
    print(f'Epoch: {epoch+1}  Training Loss: {train_loss:.4f}  Validation Loss: {val_loss:.4f}  Validation Accuracy: {val_accuracy:.4f}')

# Save the trained model
if not os.path.exists("./output"):
    os.makedirs("./output")
torch.save(model.state_dict(), "./output/emotion_detection_model_grayscale.pth")

# Plot the training and validation losses and accuracy
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracy_list, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

if not os.path.exists("./output"):
    os.makedirs("./output")
plt.tight_layout()
plt.savefig("./output/training_plot.png")

