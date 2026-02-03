"""
PSEUDOCODE / STEPS:
1. Define Helper Function 'xywhn2xyxy': 
   - Convert normalized bounding box format (x,y,w,h) to pixel coordinates (x1,y1,x2,y2).
2. Define Main Process 'run_herding':
    2.1. Define Configuration (Model path, Data dirs, Classes, Memory Size K=2000).
    2.2. Verify model existence.
    2.3. Initialize Feature Extractor and FAISS Manager.
    2.4. Stage 1: Scan Dataset.
         2.4.1. Iterate through all label files in Task 1.
         2.4.2. Verify corresponding image exists.
         2.4.3. Parse labels and store objects belonging to target classes (0-9).
    2.5. Stage 2: Extract Features.
         2.5.1. Loop exactly once through each image to extract features efficiently.
         2.5.2. Crop objects from image based on bounding boxes.
         2.5.3. Run batch feature extraction on crops.
         2.5.4. Store features and metadata (class, box) in temporary dictionary.
    2.6. Stage 3: Herding and Indexing (iCaRL strategy).
         2.6.1. Iterate through each class.
         2.6.2. Calculate Mean of all features for the class.
         2.6.3. Normalize all features.
         2.6.4. Iteratively select exemplars that best approximate the class mean.
         2.6.5. Add selected exemplars to FAISS index.
    2.7. Save the populated FAISS index to disk.
3. Main Entry Point:
   - Execute the herding process.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.utils.feature_extractor import FeatureExtractor
from src.utils.faiss_manager import FaissManager

# 1. Define Helper Function
def xywhn2xyxy(x, y, w, h, W, H):
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H)
    return [x1, y1, x2, y2]

# 2. Define Main Process
def run_herding():
    # 2.1. Define Configuration
    MODEL_PATH = 'models/model_t1.pt'
    DATA_DIR = Path('data/VOC_T1/train/labels')
    IMG_DIR = Path('data/VOC_T1/train/images')
    CLASSES = list(range(0, 10))
    K = 2000
    EXEMPLARS_PER_CLASS = K // len(CLASSES)
    
    # 2.2. Verify Model
    if not os.path.exists(MODEL_PATH):
        print("Model T1 not found. Please train it first.")
        return

    # 2.3. Initialize Components
    print("Initializing Feature Extractor...")
    extractor = FeatureExtractor(MODEL_PATH)
    manager = FaissManager()
    
    # 2.4. Stage 1: Scan Dataset
    print("Scanning dataset...")
    image_objects = {} # img_path -> list of (cls_id, box_norm)
    
    # 2.4.1. Iterate label files
    label_files = list(DATA_DIR.glob("*.txt"))
    for lf in tqdm(label_files):
        img_name = lf.stem + ".jpg"
        img_path = IMG_DIR / img_name
        # 2.4.2. Verify image existence
        if not img_path.exists():
             img_path = IMG_DIR / (lf.stem + ".jpeg")
        if not img_path.exists(): continue
        img_path = str(img_path)
        
        # 2.4.3. Parse labels
        with open(lf, 'r') as f:
            lines = f.readlines()
            
        objs = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id in CLASSES:
                box_norm = list(map(float, parts[1:]))
                objs.append((cls_id, box_norm))
        
        if objs:
            image_objects[img_path] = objs

    # 2.5. Stage 2: Extract Features
    print("Extracting features...")
    class_data = {c: {'feats': [], 'meta': []} for c in CLASSES}
    
    # 2.5.1. Loop through images
    for img_path, objs in tqdm(image_objects.items()):
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        
        crops = []
        valid_indices = []
        
        # 2.5.2. Crop objects
        for i, (cls_id, box_norm) in enumerate(objs):
            box_pixel = xywhn2xyxy(*box_norm, W, H)
            x1, y1, x2, y2 = box_pixel
            
            # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crops.append(crop)
                valid_indices.append(i)
                
        if not crops: continue
        
        # 2.5.3. Batch extract
        feats = extractor.extract_batch(crops)
        
        # 2.5.4. Store features
        for k, idx in enumerate(valid_indices):
            cls_id, box_norm = objs[idx]
            box_pixel = xywhn2xyxy(*box_norm, W, H) # Recompute or store
            
            if k < len(feats):
                class_data[cls_id]['feats'].append(feats[k])
                class_data[cls_id]['meta'].append({
                    'path': img_path,
                    'cls': cls_id,
                    'box': box_pixel
                })

    extractor.close()

    # 2.6. Stage 3: Herding and Indexing
    print("Performing Herding...")
    # 2.6.1. Iterate each class
    for cls_id in CLASSES:
        feats = np.array(class_data[cls_id]['feats'])
        metas = class_data[cls_id]['meta']
        n_samples = len(feats)
        
        if n_samples == 0:
            print(f"Warning: No objects found for class {cls_id}")
            continue
            
        print(f"Class {cls_id}: {n_samples} candidates. Selecting {EXEMPLARS_PER_CLASS}...")
        
        # 2.6.2. Calculate Mean
        mu = np.mean(feats, axis=0)
        mu = mu / np.linalg.norm(mu) # Normalize? iCaRL usually uses normalized features.
        # 2.6.3. Normalize all features
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        
        # 2.6.4. Iteratively select exemplars
        selected_indices = []
        selected_sum = np.zeros_like(mu)
        
        # We want to select k exemplars
        target_k = min(EXEMPLARS_PER_CLASS, n_samples)
        
        # Mask for available indices
        available = np.ones(n_samples, dtype=bool)
        
        for k in range(target_k):
            # Compute mean of current selection + candidate
            # We want to minimize || mu - (current_sum + x) / (k+1) ||
            # Equivalent to minimizing || (k+1)*mu - current_sum - x ||
            
            target = (k + 1) * mu - selected_sum
            
            # Find x closest to target among available
            # Dist = || target - x ||^2
            # We can use dot product if normalized? ||a-b||^2 = a^2 + b^2 - 2ab
            # Just use L2 distance
            
            # Get available features
            # This loop can be slow if n_samples is large.
            # Vectorized approach:
            
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0: break
            
            cands = feats_norm[avail_indices]
            dists = np.linalg.norm(cands - target, axis=1)
            best_idx_local = np.argmin(dists)
            best_idx = avail_indices[best_idx_local]
            
            selected_indices.append(best_idx)
            selected_sum += feats_norm[best_idx]
            available[best_idx] = False
            
        # 2.6.5. Add to FAISS
        selected_feats = feats[selected_indices] # Store original features? Or normalized?
        # FAISS usually stores raw features.
        selected_metas = [metas[i] for i in selected_indices]
        
        manager.add_vectors(selected_feats, selected_metas)

    # 2.7. Save Index
    manager.save()
    print("Herding complete. Memory populated.")

# 3. Main Entry Point
if __name__ == '__main__':
    run_herding()
