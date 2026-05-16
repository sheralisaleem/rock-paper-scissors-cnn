import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# Model Architecture Definition (Imported from Training Phase)
# ---------------------------------------------------------
class DeepRPS(nn.Module):
    def __init__(self):
        super(DeepRPS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(512 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 3) 

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, 512 * 9 * 9)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
# ImageFolder sorts alphabetically: 0: paper, 1: rock, 2: scissors
classes = ['paper', 'rock', 'scissors'] 

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Model
model = DeepRPS()
try:
    model.load_state_dict(torch.load("deep_rps_model.pth", map_location=device, weights_only=True))
    print("Successfully loaded model weights from deep_rps_model.pth")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

model.to(device)
model.eval()

# Preprocessing Pipeline (Identical to training validation transform)
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------------------------------------------------
# OpenCV Webcam Feed
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

# Define ROI size (the crop size before resizing)
roi_size = 300

print("Starting Webcam. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break
        
    # Flip the frame horizontally for a more natural mirror view
    frame = cv2.flip(frame, 1)
    
    height, width, _ = frame.shape
    
    # Calculate ROI coordinates (Center crop)
    x1 = int((width - roi_size) / 2)
    y1 = int((height - roi_size) / 2)
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    
    # Draw ROI rectangle on the original frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place Hand Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Extract ROI
    roi_bgr = frame[y1:y2, x1:x2]
    
    if roi_bgr.shape[0] > 0 and roi_bgr.shape[1] > 0:
        # Convert BGR to RGB for PyTorch/PIL
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for torchvision transforms
        pil_image = Image.fromarray(roi_rgb)
        
        # Preprocess exactly like the DataLoader
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(device) # Create a mini-batch of size 1
        
        # Inference
        with torch.no_grad():
            outputs = model(input_batch)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_idx = predicted.item()
            conf_score = confidence.item() * 100
            
            pred_class = classes[class_idx]
            
            # Display Prediction and Confidence
            text = f"Prediction: {pred_class.upper()} ({conf_score:.1f}%)"
            
            # Dynamic text color based on confidence score
            if conf_score > 80:
                color = (0, 255, 0) # Green (High Confidence)
            elif conf_score > 50:
                color = (0, 255, 255) # Yellow (Medium Confidence)
            else:
                color = (0, 0, 255) # Red (Low Confidence)
                
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show the live feed
    cv2.imshow('Deep-RPS Live Game', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
