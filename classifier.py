import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
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

    # Initialize variables for tracking loss and accuracy
    train_loss_history = []
    test_accuracy_history = []

    # Training the model
    num_epochs = 30
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
        train_loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluate the model after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracy_history.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), './resnet_model.pth')
    print("Model saved as resnet_model.pth")

    # Visualization: Plot Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracy_history, label='Test Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Confusion Matrix and Classification Report
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Generate Classification Report
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions, target_names=train_data.classes))
