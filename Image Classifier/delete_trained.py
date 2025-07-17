import os

dataset_dir = "S:/vscode/vscode py files/Image Classifier/dataset"
model_save_path = "S:/vscode/vscode py files/Image Classifier/model/model.pth"
class_indices_path = "S:/vscode/vscode py files/Image Classifier/model/class_indices.json"
TRAINED_MARKER = "trained.txt"

print("Deleting existing trained data...")
deleted_count = 0

# Delete trained.txt files from dataset subfolders
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        marker_path = os.path.join(folder_path, TRAINED_MARKER)
        if os.path.exists(marker_path):
            try:
                os.remove(marker_path)
                print(f" - Removed {marker_path}")
                deleted_count += 1
            except Exception as e:
                print(f" - Error removing {marker_path}: {e}")

# Delete model file
if os.path.exists(model_save_path):
    try:
        os.remove(model_save_path)
        print(f" - Removed {model_save_path}")
        deleted_count += 1
    except Exception as e:
        print(f" - Error removing {model_save_path}: {e}")

# Delete class indices file
if os.path.exists(class_indices_path):
    try:
        os.remove(class_indices_path)
        print(f" - Removed {class_indices_path}")
        deleted_count += 1
    except Exception as e:
        print(f" - Error removing {class_indices_path}: {e}")

if deleted_count == 0:
    print("No trained data files found to delete.")
else:
    print(f"Deleted {deleted_count} trained data files.")
