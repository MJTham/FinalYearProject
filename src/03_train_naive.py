"""
PSEUDOCODE / STEPS:
1. Initialize the training process for Task 2 (Naive approach):
    1.1. Print start message.
    1.2. Verify that the previous model (model_t1.pt) exists; error if missing.
    1.3. Load the Task 1 model as the starting point.
    1.4. Define a callback function to clear MPS cache after each epoch (memory management).
    1.5. Attach the callback to the model.
    1.6. Check if there is an existing checkpoint to resume training.
    1.7. If checkpoint exists, resume training from there.
    1.8. If no checkpoint, start fresh training on Task 2 data (VOC_T2.yaml):
         - Use same parameters as Task 1 (5 epochs, 640px, MPS device).
    1.9. Define source and destination paths for the best model weights.
    1.10. Save/Copy the best weights to 'models/model_t2_naive.pt'.
    1.11. Handle error if weights are not found.
2. Main Entry Point:
    2.1. Execute the naive training function.
"""

from ultralytics import YOLO
import shutil
import os
import torch
import gc

# 1. Initialize Training Process for Task 2 (Naive)
def train_naive():
    # 1.1. Print start message
    print("Starting Naive T2 Training...")
    
    # 1.2. Verify prior model existence
    if not os.path.exists('models/model_t1.pt'):
        print("Error: models/model_t1.pt not found. Run train_baseline.py first.")
        return

    # 1.3. Load the T1 model
    model = YOLO('models/model_t1.pt')

    # 1.4. Define memory cleanup callback
    def on_train_epoch_end(trainer):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            gc.collect()
            print("MPS cache cleared and garbage collected.")

    # 1.5. Attach callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 1.6. Check for checkpoint
    checkpoint_path = 'models/model_t2_naive/weights/last.pt'
    if os.path.exists(checkpoint_path):
        # 1.7. Resume training
        print(f"Resuming training from {checkpoint_path} with batch=8...")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True, batch=8)
    else:
        # 1.8. Start fresh training
        # Train the model on T2 data (fresh start)
        results = model.train(
            data='data/VOC_T2.yaml',
            epochs=5,
            imgsz=640,
            project='models',
            name='model_t2_naive',
            exist_ok=True,
            patience=5,
            device='mps',
            workers=0
        )
    
    # 1.9. Define save paths
    src_path = 'models/model_t2_naive/weights/best.pt'
    dst_path = 'models/model_t2_naive.pt'
    
    # 1.10. Save model weights
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Saved Naive T2 model to {dst_path}")
    # 1.11. Handle errors
    else:
        print("Error: Naive T2 model weights not found.")

# 2. Main Entry Point
if __name__ == '__main__':
    train_naive()
