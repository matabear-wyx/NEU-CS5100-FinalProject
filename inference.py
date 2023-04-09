import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import model

# Define data transforms for input images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define class labels
classes = ["Angry", "Fear", "Happy", "Sadness", "Surprise"]

# Load the trained model
model = model.EmotionDetector()
model.load_state_dict(torch.load("./output/emotion_detection_model_grayscale_best.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Open the camera device
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera device")
    exit()

# Process each frame from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale and detect faces using a Haar cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a border around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face region and preprocess it
        face_img = gray[y:y+h, x:x+w]
        pil_img = Image.fromarray(face_img)
        img_tensor = data_transforms(pil_img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        # Classify the face using the trained model
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)
            confidence = np.exp(outputs[0][predicted].cpu().numpy()) / np.exp(outputs[0].cpu().numpy()).sum()

        # Show the predicted class and confidence level on the border of the face
        cv2.putText(frame, f"{classes[predicted]} ({float(confidence):.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Show the processed frame in the OpenCV window
    cv2.imshow("Emotion Detection", frame)

    # Wait for a key press and check if the user wants to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera device and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
