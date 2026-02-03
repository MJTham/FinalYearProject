"""
PSEUDOCODE / STEPS:
1. Define 'FeatureExtractor' Class:
   - Wrapper around YOLOv8 model to extract intermediate layer features.
   - 1.1. Initialization (__init__):
          - Load the YOLO model.
          - Identify target layer index (default 9, usually SPPF).
          - Register the forward hook.
   - 1.2. _register_hook:
          - Attaches a PyTorch forward hook to the specific layer to intercept output.
   - 1.3. _hook_fn:
          - Callback function that stores layer output during valid inference.
   - 1.4. extract_batch:
          - Runs inference on a batch of images (crops).
          - Aggregates the intercepted feature maps.
          - Performs Global Average Pooling to flatten features into vectors (1D).
   - 1.5. extract_one:
          - Helper to extract features for a single object crop given bounding box.
   - 1.6. close:
          - Cleanly removes the hook.
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Define FeatureExtractor Class
class FeatureExtractor:
    # 1.1. Initialization
    def __init__(self, model_path, layer_index=9):
        self.model = YOLO(model_path)
        self.layer_index = layer_index
        self.features = []
        self.hook_handle = None
        self._register_hook()

    # 1.2. Register Hook
    def _register_hook(self):
        # Access the underlying PyTorch model
        # model.model is the DetectionModel
        # model.model.model is the nn.Sequential list of layers
        # Layer 9 is usually SPPF in YOLOv8n
        try:
            target_layer = self.model.model.model[self.layer_index]
            self.hook_handle = target_layer.register_forward_hook(self._hook_fn)
            print(f"Hook registered on layer {self.layer_index}: {target_layer}")
        except Exception as e:
            print(f"Error registering hook: {e}")

    # 1.3. Hook Function
    def _hook_fn(self, module, input, output):
        # Output is [Batch, Channels, H, W]
        self.features.append(output)

    # 1.4. Extract Batch
    def extract_batch(self, images):
        # images: list of numpy arrays (crops)
        self.features = []
        # Run inference using predict
        self.model.predict(images, verbose=False, imgsz=224) 
        
        # Aggregate features
        batch_features = []
        for feat_map in self.features:
            # feat_map: [B, C, H, W] -> Global Average Pooling -> [B, C]
            pooled = torch.mean(feat_map, dim=[2, 3]).cpu().numpy()
            batch_features.append(pooled)
            
        if batch_features:
            return np.vstack(batch_features)
        return np.array([])

    # 1.5. Extract Single Object
    def extract_one(self, img_path, box):
        # box: [x1, y1, x2, y2]
        img = cv2.imread(str(img_path))
        if img is None: return None
        x1, y1, x2, y2 = map(int, box)
        
        # Clamp box to image
        h, w = img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1: return None
        
        crop = img[y1:y2, x1:x2]
        
        self.features = []
        self.model.predict(crop, verbose=False, imgsz=224)
        
        if not self.features: return None
        
        feat_map = self.features[0] # [1, C, H, W]
        feat_vec = torch.mean(feat_map, dim=[2, 3]).squeeze().cpu().numpy()
        return feat_vec

    # 1.6. Cleanup
    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()
