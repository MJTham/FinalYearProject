"""
PSEUDOCODE / STEPS:
1. Define 'create_replay_dataset' Function:
   - Configures and creates a new dataset combining Task 2 data with Task 1 exemplars.
   - Arguments: Path to Task 2 YAML, Output directory path.
   - 1.1. Setup Destination Directories (images/labels).
   - 1.2. Load Task 2 Configuration:
          - Parse YAML to find image and label paths.
          - Copy (Symlink) Task 2 training images and labels to the new destination.
   - 1.3. Load Exemplars:
          - Initialize 'FaissManager' to access stored memory.
          - Retrieve all exemplar metadata.
          - Iterate through exemplars and Copy (Symlink) their original images and labels to the new destination.
          - Ensure no duplicates are added.
   - 1.4. Generate New YAML Configuration:
          - Create a new 'VOC_T2_Continual.yaml' file pointing to this mixed dataset.
          - Ensure class names and counts match the full dataset (20 classes).
   - 1.5. Return path to the new YAML file.
2. Main Entry Point:
   - Execution block to run the function directly if needed.
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm
from src.utils.faiss_manager import FaissManager

# 1. Define 'create_replay_dataset' functionality
def create_replay_dataset(t2_yaml_path='data/VOC_T2.yaml', output_dir='data/VOC_T2_Continual'):
    print("Creating Replay Dataset...")
    
    # 1.1. Setup Directories
    output_dir = Path(output_dir)
    images_dir = output_dir / 'train' / 'images'
    labels_dir = output_dir / 'train' / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 1.2. Load and Copy Task 2 Data
    with open(t2_yaml_path, 'r') as f:
        t2_config = yaml.safe_load(f)
        
    t2_base = Path(t2_config['path'])
    t2_train_img = t2_base / t2_config['train']
    t2_train_lbl = t2_train_img.parent / 'labels' 
    
    print("Copying T2 data...")
    for img_file in tqdm(list(t2_train_img.glob("*.*"))):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']: continue
        
        # Link Image
        dst_img = images_dir / img_file.name
        if not dst_img.exists():
            os.symlink(img_file.resolve(), dst_img)
            
        # Link Label
        lbl_file = t2_train_lbl / (img_file.stem + ".txt")
        if lbl_file.exists():
            dst_lbl = labels_dir / lbl_file.name
            if not dst_lbl.exists():
                os.symlink(lbl_file.resolve(), dst_lbl)

    # 1.3. Load and Copy Exemplars (Memory)
    print("Loading Exemplars from Memory...")
    manager = FaissManager()
    exemplars = manager.get_all_exemplars()
    
    seen_exemplars = set()
    
    print("Adding Exemplars...")
    for ex in tqdm(exemplars):
        img_path = Path(ex['path'])
        if img_path in seen_exemplars: continue
        seen_exemplars.add(img_path)
        
        # Link Exemplar Image
        dst_img = images_dir / img_path.name
        if not dst_img.exists():
            os.symlink(img_path.resolve(), dst_img)
            
        # Link Exemplar Label (Original T1 label)
        lbl_path = img_path.parent.parent / 'labels' / (img_path.stem + ".txt")
        
        if lbl_path.exists():
            dst_lbl = labels_dir / lbl_path.name
            if not dst_lbl.exists():
                os.symlink(lbl_path.resolve(), dst_lbl)
                
    # 1.4. Generate New YAML Config
    new_yaml = output_dir.parent / 'VOC_T2_Continual.yaml'
    data_config = {
        'path': str(output_dir.parent), # Base dir
        'train': 'VOC_T2_Replay/train/images',
        'val': t2_config['val'], # Validate on T2 validation set
        'nc': 20,
        'names': t2_config['names']
    }
    
    # Fix absolute path for validation
    val_path = t2_base / t2_config['val']
    data_config['val'] = str(val_path)
    
    with open(new_yaml, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
        
    print(f"Replay dataset created at {output_dir}")
    # 1.5. Return Path
    return str(new_yaml)

# 2. Main Entry Point
if __name__ == '__main__':
    create_replay_dataset()
