import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18  # Example CNN architecture
import os
import torch.nn as nn

# Ensure the model file exists
model_path = './resnet_model.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Train or place the file in the correct directory.")
    exit()

# Load the model used during training
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Initialize the model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 24  # Update based on your class count
model = ResNetModel(num_classes=num_classes)
model.load_state_dict(torch.load('./resnet_model.pth', map_location=device))
model.eval()
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert OpenCV image (NumPy array) to PIL format
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Use class names from the trained model
labels_dict = {i: label for i, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])}  # Your 24 labels

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Preprocess the frame for prediction
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    try:
        frame_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
        frame_tensor = frame_tensor.to(device)

        # Predict gesture using the CNN
        with torch.no_grad():
            prediction = model(frame_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()

        # Display the predicted label on the frame
        predicted_label = labels_dict[predicted_class]
        cv2.putText(frame, f"Prediction: {predicted_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error during prediction: {e}")

    # Convert RGB back to BGR for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
