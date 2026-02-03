"""
PSEUDOCODE / STEPS:
1. Define Configuration Constants (Paths, Classes, Class Names).
2. Define the 'process_dataset' function that splits data into Task 1 and Task 2:
    2.1. Define source and destination directories for images and labels.
    2.2. Create destination directories if they don't exist.
    2.3. Check if the source label directory exists; skip if missing.
    2.4. Loop through every label file in the source directory:
         2.4.1. Read all lines (objects) from the label file.
         2.4.2. Filter lines: Check if class ID belongs to Task 1 or Task 2.
         2.4.3. Find the corresponding image file (checking multiple extensions).
         2.4.4. If the image exists, save the filtered label file to the respective Task folder.
         2.4.5. Create a symbolic link for the image in the Task folder (to save space).
3. Define the 'create_yaml' function to generate YOLO dataset configuration files.
4. Main Execution Block:
    4.1. Process both 'train' and 'valid' splits.
    4.2. Generate YAML files for Task 1 and Task 2.
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# 1. Define Configuration Constants
BASE_DIR = Path("/Users/mjtham/Documents/Yolov8ContinualLearning/data")
SOURCE_DIR = BASE_DIR / "VOC"
T1_DIR = BASE_DIR / "VOC_T1"
T2_DIR = BASE_DIR / "VOC_T2"

T1_CLASSES = list(range(0, 10))
T2_CLASSES = list(range(10, 20))

NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 2. Define process_dataset function
def process_dataset(split):
    # 2.1. Define source and destination directories
    src_images = SOURCE_DIR / split / "images"
    src_labels = SOURCE_DIR / split / "labels"
    
    t1_images = T1_DIR / split / "images"
    t1_labels = T1_DIR / split / "labels"
    t2_images = T2_DIR / split / "images"
    t2_labels = T2_DIR / split / "labels"
    
    # 2.2. Create destination directories
    for p in [t1_images, t1_labels, t2_images, t2_labels]:
        p.mkdir(parents=True, exist_ok=True)
    
    # 2.3. Check source existence
    if not src_labels.exists():
        print(f"Warning: {src_labels} does not exist. Skipping {split}.")
        return

    # 2.4. Loop through every label file
    for label_file in tqdm(list(src_labels.glob("*.txt")), desc=f"Processing {split}"):
        
        # 2.4.1. Read all lines
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        t1_lines = []
        t2_lines = []
        
        # 2.4.2. Filter lines by class ID
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            
            if cls_id in T1_CLASSES:
                t1_lines.append(line)
            elif cls_id in T2_CLASSES:
                t2_lines.append(line)
                
        # 2.4.3. Find corresponding image file
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            temp = src_images / (label_file.stem + ext)
            if temp.exists():
                image_file = temp
                break
        
        if not image_file:
             continue # Skip if image not found

        # 2.4.4. Save filtered labels and 2.4.5. link images for Task 1
        if t1_lines:
            with open(t1_labels / label_file.name, 'w') as f:
                f.writelines(t1_lines)
            # Symlink image
            dst_img = t1_images / image_file.name
            if not dst_img.exists():
                os.symlink(image_file, dst_img)
                
        # 2.4.4. Save filtered labels and 2.4.5. link images for Task 2
        if t2_lines:
            with open(t2_labels / label_file.name, 'w') as f:
                f.writelines(t2_lines)
            # Symlink image
            dst_img = t2_images / image_file.name
            if not dst_img.exists():
                os.symlink(image_file, dst_img)

# 3. Define create_yaml function
def create_yaml(name, path_dir):
    data = {
        'path': str(path_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 20, 
        'names': NAMES
    }
    with open(BASE_DIR / f"{name}.yaml", 'w') as f:
        yaml.dump(data, f, sort_keys=False)

# 4. Main Execution Block
if __name__ == "__main__":
    print("Preparing T1 and T2 datasets...")
    
    # 4.1. Process both splits
    process_dataset("train")
    process_dataset("valid")
    
    # 4.2. Generate YAML files
    create_yaml("VOC_T1", T1_DIR)
    create_yaml("VOC_T2", T2_DIR)
    print("Data preparation complete.")
