import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your dataset directory
dataset_dir = "dataset"

# Subdirectories for train and test data
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio
train_split = 0.8

# Valid image extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Iterate over each class directory in the dataset
for class_name in os.listdir(dataset_dir):
    source_class_dir = os.path.join(dataset_dir, class_name)

    if not os.path.isdir(source_class_dir):  # Skip non-directory files
        continue

    # Get all image files in the current class directory
    image_files = [f for f in os.listdir(source_class_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:  # Skip empty folders
        print(f"Skipping empty folder: {source_class_dir}")
        continue

    print(f"Processing class: {class_name}, found {len(image_files)} images.")

    # Split into train and test
    train_files, test_files = train_test_split(image_files, test_size=1 - train_split, shuffle=True)

    # Create class subdirectories in train and test folders
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move files to train directory
    for file in train_files:
        shutil.copy(os.path.join(source_class_dir, file), os.path.join(train_class_dir, file))

    # Move files to test directory
    for file in test_files:
        shutil.copy(os.path.join(source_class_dir, file), os.path.join(test_class_dir, file))

print("Dataset restructured successfully!")
