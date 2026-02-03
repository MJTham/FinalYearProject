"""
PSEUDOCODE / STEPS:
1. Initialize the Continual Training Process (Experience Replay):
    1.1. Print start message.
    1.2. Generate the "Replay Dataset":
         - This involves mixing new Task 2 data with a subset of old Task 1 data (exemplars).
         - Returns the path to the new YAML configuration file.
    1.3. Verify that the previous model (Task 1) exists.
    1.4. Load the Task 1 model as the starting point.
    1.5. Train the model on the Mixed Dataset:
         - Uses the same robust parameters (MPS device, no workers, 5 epochs).
         - This ensures the model learns Task 2 while "revisiting" Task 1.
    1.6. Define source and destination paths for the best weights.
    1.7. Save/Copy the best weights to 'models/model_t2_continual.pt'.
    1.8. Handle error if weights are not found.
2. Main Entry Point:
    2.1. Execute the continual training function.
"""

from ultralytics import YOLO
from src.utils.custom_dataloader import create_replay_dataset
import shutil
import os

# 1. Initialize Continual Training
def train_continual():
    # 1.1. Print start message
    print("Starting Continual Training (Experience Replay)...")
    
    # 1.2. Generate Dataset
    # This step is critical: it creates the 'memory' mix
    dataset_yaml = create_replay_dataset()
    
    # 1.3. Verify T1 Model
    if not os.path.exists('models/model_t1.pt'):
        print("Error: models/model_t1.pt not found. Please run train_baseline.py first.")
        return

    # 1.4. Load T1 Model
    model = YOLO('models/model_t1.pt')
    
    # 1.5. Train on the mixed dataset
    results = model.train(
        data=dataset_yaml,
        epochs=5,
        imgsz=640,
        project='models',
        name='model_t2_continual',
        exist_ok=True,
        patience=5,
        device='mps',
        workers=0
    )
    
    # 1.6. Define save paths
    src_path = 'models/model_t2_continual/weights/best.pt'
    dst_path = 'models/model_t2_continual.pt'
    
    # 1.7. Save model
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Saved Continual model to {dst_path}")
    # 1.8. Handle errors
    else:
        print("Error: Continual model weights not found.")

# 2. Main Entry Point
if __name__ == '__main__':
    train_continual()
