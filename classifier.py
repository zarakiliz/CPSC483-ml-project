import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Define the model architecture using ResNet18
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Function to plot classification report
def plot_classification_report(report_text):
    plt.figure(figsize=(10, 6))
    plt.text(0.01, 0.05, str(report_text), {'fontsize': 12}, fontproperties='monospace') 
    plt.axis('off')
    plt.title("Classification Report")
    plt.show()

# Ensure safe multiprocessing (Windows-specific)
if __name__ == '__main__':
    # Load datasets
    train_data = datasets.ImageFolder(root='dataset/train', transform=transform)
    test_data = datasets.ImageFolder(root='dataset/test', transform=transform)

    # Create data loaders
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    # Print class mapping
    print("Class Mapping:", train_data.class_to_idx)

    # Initialize the ResNet model
    num_classes = len(train_data.classes)
    model = ResNetModel(num_classes)

    # Move the model to the appropriate device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) 

    # Define class weights (if you want to use weighted loss)
    class_weights = torch.tensor([1.0] * num_classes).to(device) 

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Visualization of the model architecture
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Dummy input for visualization
    torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True)
    print("Model exported to resnet18.onnx")

    # Training the model
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluating the model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and true labels
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), './resnet_model.pth')
    print("Model saved as resnet_model.pth")

    # Generate and visualize the confusion matrix
    class_names = train_data.classes
    plot_confusion_matrix(all_labels, all_predictions, class_names)

    # Generate and visualize the classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(report)
    plot_classification_report(report)
