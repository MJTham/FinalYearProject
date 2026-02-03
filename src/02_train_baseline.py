"""
PSEUDOCODE / STEPS:
1. Initialize the training process for Request 'Task 1':
    1.1. Print start message.
    1.2. Load the base YOLOv8 model (yolov8n.pt) as a starting point.
    1.3. Train the model on Task 1 data (VOC_T1.yaml) for 5 epochs.
         - Configure parameters: 640px image size, MPS device, 0 workers (for stability).
         - Save results to 'models/model_t1'.
    1.4. Locate the best weights from the training run.
    1.5. Copy/Save the best weights to the standard path 'models/model_t1.pt'.
    1.6. Handle error if weights are not found.
2. Main Entry Point:
    2.1. Execute the training function.
"""

from ultralytics import YOLO
import shutil
import os

# 1. Initialize Training Process
def train_t1():
    # 1.1. Print start message
    print("Starting Baseline T1 Training...")
    
    # 1.2. Load base model
    model = YOLO('models/yolov8n.pt')  # load a pretrained model (recommended for training)

    # 1.3. Train the model
    results = model.train(
        data='data/VOC_T1.yaml',
        epochs=5,
        imgsz=640,
        project='models',
        name='model_t1',
        exist_ok=True,
        patience=5,
        device='mps',
        workers=0
    )
    
    # 1.4. Locate best weights
    src_path = 'models/model_t1/weights/best.pt'
    dst_path = 'models/model_t1.pt'
    
    # 1.5. Save model weights
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Saved T1 model to {dst_path}")
    # 1.6. Handle errors
    else:
        print("Error: T1 model weights not found.")

# 2. Main Entry Point
if __name__ == '__main__':
    train_t1()
