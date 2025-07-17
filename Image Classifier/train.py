import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import json
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define paths
dataset_dir = "S:/vscode/vscode py files/Image Classifier/dataset"
model_save_path = "S:/vscode/vscode py files/Image Classifier/model/model.pth"
class_indices_path = "S:/vscode/vscode py files/Image Classifier/model/class_indices.json"
weights_path = "C:/Users/yuvar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"  # Update if different
TRAINED_MARKER = "trained.txt"

# Image parameters
img_height, img_width = 224, 224
batch_size = 16
num_epochs = 20  # Increased from 10

def check_dataset_folders(dataset_dir):
    print("\nChecking dataset folders...")
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return None, 0
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    folder_count = len(folders)
    print(f"Found {folder_count} folders in the dataset directory '{dataset_dir}':")
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"Warning: Folder '{folder}' contains no valid images (.jpg, .jpeg, .png).")
        else:
            print(f" - {folder}")
    return folders, folder_count

def check_trained_folders(dataset_dir, folders):
    print("Checking trained folders...")
    trained_folders = []
    untrained_folders = []
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        marker_path = os.path.join(folder_path, TRAINED_MARKER)
        if os.path.exists(marker_path):
            trained_folders.append(folder)
        else:
            untrained_folders.append(folder)
    return trained_folders, untrained_folders

def count_files_in_folders(dataset_dir, folders):
    print("Counting files in folders...")
    print("\nFile count in each folder:")
    total_files = 0
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        file_count = len(files)
        total_files += file_count
        print(f" - {folder}: {file_count} files")
    return total_files

def mark_folder_as_trained(folder_path):
    print(f"Marking folder as trained: {folder_path}")
    marker_path = os.path.join(folder_path, TRAINED_MARKER)
    with open(marker_path, 'w') as f:
        f.write("This folder has been trained.")

def select_folders_to_train(folders):
    print("Selecting folders to train...")
    print("\nAvailable folders:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    print("\nEnter the numbers of the folders you want to train (e.g., '1,3,5'), or 'all' to train all folders:")
    choice = input().strip().lower()
    if choice == 'all':
        return folders
    try:
        indices = [int(i) - 1 for i in choice.split(',')]
        selected_folders = [folders[i] for i in indices if 0 <= i < len(folders)]
        return selected_folders
    except (ValueError, IndexError):
        print("Invalid input. Selecting all folders by default.")
        return folders

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check dataset folders
folders, folder_count = check_dataset_folders(dataset_dir)
if not folders:
    exit()

# Check trained and untrained folders
trained_folders, untrained_folders = check_trained_folders(dataset_dir, folders)

# Display trained and untrained folders
if trained_folders:
    print("\nAlready trained folders:")
    for folder in trained_folders:
        print(f" - {folder}")
else:
    print("\nNo folders have been trained yet.")

if untrained_folders:
    print("\nNew (untrained) folders:")
    for folder in untrained_folders:
        print(f" - {folder}")
else:
    print("\nNo new folders to train.")

# Prompt to continue
input("\nPress Enter to continue...")

# Select folders to train
all_folders = folders
if trained_folders:
    retrain = input("\nDo you want to retrain already trained folders? (y/n): ").strip().lower()
    if retrain != 'y':
        all_folders = untrained_folders

if not all_folders:
    print("\nNo folders available to train.")
    exit()

total_files = count_files_in_folders(dataset_dir, all_folders)
print(f"\nTotal images to train: {total_files}")
folders_to_train = select_folders_to_train(all_folders)
if not folders_to_train:
    print("No folders selected for training. Exiting.")
    exit()

print(f"\nSelected folders to train: {', '.join(folders_to_train)}")
response = input("\nDo you want to train these folders? (y/n): ").strip().lower()
if response != 'y':
    print("Training aborted.")
    exit()

# Data augmentation and preprocessing
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),  # Added random crop
    transforms.RandomRotation(30),  # Increased from 20
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Added
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Enhanced
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
try:
    full_dataset = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    # Filter dataset to include only selected folders
    selected_indices = [i for i, (path, _) in enumerate(full_dataset.samples) if os.path.basename(os.path.dirname(path)) in folders_to_train]
    filtered_dataset = torch.utils.data.Subset(full_dataset, selected_indices)
    
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images belonging to {len(folders_to_train)} classes.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Ensure the selected folders contain valid images.")
    exit()

# Get class indices
class_indices = {k: v for k, v in full_dataset.class_to_idx.items() if k in folders_to_train}
print("Class indices:", class_indices)

# Save class indices
try:
    os.makedirs(os.path.dirname(class_indices_path), exist_ok=True)
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved to {class_indices_path}")
except Exception as e:
    print(f"Error saving class indices: {e}")
    exit()

# Build the model
try:
    model = models.resnet18(weights=None).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Freeze all layers except the last few
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add 50% dropout
        nn.Linear(num_ftrs, len(class_indices))
    ).to(device)
except Exception as e:
    print(f"Error loading model or weights: {e}")
    print(f"Ensure {weights_path} exists and is accessible.")
    exit()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)  # Reduced from 0.001

# Training loop
try:
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {running_loss / 10:.4f}")
                running_loss = 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# Save the model
try:
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

# Mark selected folders as trained
for folder in folders_to_train:
    mark_folder_as_trained(os.path.join(dataset_dir, folder))