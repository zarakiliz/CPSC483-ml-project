# this file trains the classifier with Pytorch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define transformations (resize, normalize, etx.)/ data augmentation
transform = transforms.Compose([
    transforms.Resize((224,224)), # resize all images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
])

# Load train and test datasets
train_data = datasets.ImageFolder(root='dataset/train', transform=transform)
test_data = datasets.ImageFolder(root='dataset/test', transform=transform)

# create dataloaders
# dataloaders manage and load data during training and testing. They handle batching, 
# shuffling and parallel data loading to make it easier to feed the model data

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# print class mapping
# class mapping helps trabslate bwtn human readable class names and numeric labels \
# required for model training

print("Class Mapping:", train_data.class_to_idx)