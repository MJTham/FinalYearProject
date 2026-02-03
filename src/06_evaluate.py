"""
PSEUDOCODE / STEPS:
1. Define 'evaluate_model' Helper Function:
   - Takes model path, task name, and dataset configuration.
   - Runs model validation using YOLOv8.
   - Extracts and returns key metrics: mAP@50, mAP@50-95, Precision, Recall.
2. Define Main Evaluation Process:
    2.1. Initialize Report.
    2.2. Evaluate Baseline Model (Task 1) on Task 1 Data.
    2.3. Evaluate Naive Model (Task 2) on both Task 1 (checking Forgetting) and Task 2 (checking Plasticity).
    2.4. Evaluate Continual Model (if it exists) on both Task 1 and Task 2.
    2.5. Calculate Forgetting Scores:
         - Forgetting = Original Performance - Current Performance on Old Task.
    2.6. Print detailed textual summary to the console.
    2.7. Construct a structured JSON dictionary containing all calculated metrics.
    2.8. Save this JSON dictionary to 'evaluation_results/metrics.json' for the Dashboard app to consume.
3. Main Entry Point:
    3.1. Execute the main evaluation function.
"""

from ultralytics import YOLO
import os

# 1. Define Helper Function
def evaluate_model(model_path, task_name, data_yaml):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return 0.0
        
    print(f"Evaluating {model_path} on {data_yaml}...")
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml, 
        project='evaluation_results', 
        name=f'{task_name}_eval',
        exist_ok=True
    )
    return {
        'map50': metrics.box.map50,
        'map50-95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr
    }

# 2. Define Main Evaluation Process
def main():
    # 2.1. Initialize Report
    print("--- Evaluation Report ---")
    
    # 2.2. Baseline T1 Performance
    metrics_t1_t1 = evaluate_model('models/model_t1.pt', 't1_on_t1', 'data/VOC_T1.yaml')
    
    # 2.3. Naive Model Performance
    metrics_naive_t1 = evaluate_model('models/model_t2_naive.pt', 'naive_on_t1', 'data/VOC_T1.yaml')
    metrics_naive_t2 = evaluate_model('models/model_t2_naive.pt', 'naive_on_t2', 'data/VOC_T2.yaml')
    
    # 2.4. Continual Model Performance (if exists)
    if os.path.exists('models/model_t2_continual.pt'):
        metrics_cont_t1 = evaluate_model('models/model_t2_continual.pt', 'continual_on_t1', 'data/VOC_T1.yaml')
        metrics_cont_t2 = evaluate_model('models/model_t2_continual.pt', 'continual_on_t2', 'data/VOC_T2.yaml')
        
        # 2.5. Calculate Forgetting (Continual)
        forgetting_cont = metrics_t1_t1['map50'] - metrics_cont_t1['map50']
    else:
        metrics_cont_t1 = {'map50': 0.0, 'map50-95': 0.0, 'precision': 0.0, 'recall': 0.0}
        metrics_cont_t2 = {'map50': 0.0, 'map50-95': 0.0, 'precision': 0.0, 'recall': 0.0}
        forgetting_cont = 0.0

    # 2.5. Calculate Forgetting (Naive)
    forgetting_naive = metrics_t1_t1['map50'] - metrics_naive_t1['map50']
    
    # 2.6. Print Summary
    print("\n--- Results Summary ---")
    print(f"Baseline T1 (T1 map50): {metrics_t1_t1['map50']:.4f}")
    print(f"Naive T2 (T1 map50):    {metrics_naive_t1['map50']:.4f} (Forgetting: {forgetting_naive:.4f})")
    print(f"Naive T2 (T2 map50):    {metrics_naive_t2['map50']:.4f}")
    
    if os.path.exists('models/model_t2_continual.pt'):
        print(f"Continual T2 (T1 map50): {metrics_cont_t1['map50']:.4f} (Forgetting: {forgetting_cont:.4f})")
        print(f"Continual T2 (T2 map50): {metrics_cont_t2['map50']:.4f}")
        
    # 2.7. Construct Data Dictionary
    import json
    results_data = {
        'Baseline': {
            'T1_Eval': metrics_t1_t1,
            'T2_Eval': None
        },
        'Naive': {
            'T1_Eval': metrics_naive_t1,
            'T2_Eval': metrics_naive_t2,
            'Forgetting': forgetting_naive
        },
        'Continual': {
            'T1_Eval': metrics_cont_t1,
            'T2_Eval': metrics_cont_t2,
            'Forgetting': forgetting_cont
        }
    }
    
    # 2.8. Save Results
    out_file = 'evaluation_results/metrics.json'
    with open(out_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"\n✅ Metrics saved to {out_file}")

# 3. Main Entry Point
if __name__ == '__main__':
    main()
