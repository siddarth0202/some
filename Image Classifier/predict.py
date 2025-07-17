import torch
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import glob

# Define paths
dataset_dir = "S:/vscode/vscode py files/Image Classifier/dataset"
untrained_dir = "S:/vscode/vscode py files/Image Classifier/untrained"  # Optional folder for new images
model_path = "S:/vscode/vscode py files/Image Classifier/model/model.pth"
class_indices_path = "S:/vscode/vscode py files/Image Classifier/model/class_indices.json"

# Image parameters
img_height, img_width = 224, 224

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load class indices
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_labels = list(class_indices.keys())
    print(f"Classes: {class_labels}")
except Exception as e:
    print(f"Error loading class indices: {e}")
    exit()

# Load the model
try:
    model = models.resnet18(weights=None).to(device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Get list of trained images (to check against uploaded images)
trained_images = []
try:
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            for file in glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.jpeg")) + glob.glob(os.path.join(folder_path, "*.png")):
                trained_images.append(os.path.normpath(file))
except Exception as e:
    print(f"Error scanning dataset: {e}")

# Initialize tkinter
root = tk.Tk()
root.withdraw()

while True:
    # Open file explorer to select a new image
    initial_dir = untrained_dir if os.path.exists(untrained_dir) else os.path.dirname(dataset_dir)
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        initialdir=initial_dir,
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not image_path:
        print("No image selected.")
        cont = messagebox.askyesno("Continue", "Do you want to select another image?")
        if not cont:
            break
        continue

    # Check if image is in trained dataset
    image_path_norm = os.path.normpath(image_path)
    if image_path_norm in trained_images:
        print(f"Error: Selected image {image_path} is part of the trained dataset.")
        cont = messagebox.askyesno("Continue", "Please select a non-trained image. Do you want to try again?")
        if not cont:
            break
        continue

    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = class_labels[predicted.item()]
            confidence = confidence.item()

        # Output the predicted animal and confidence
        print(f"The selected image is a {predicted_class} (Confidence: {confidence:.2%})")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

    # Prompt to continue
    cont = messagebox.askyesno("Continue", "Do you want to predict another image?")
    if not cont:
        break

# Cleanup
root.destroy()
print("Exiting.")