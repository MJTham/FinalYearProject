"""
PSEUDOCODE / STEPS:
1. Imports and Configuration:
   - Import necessary libraries (Streamlit, YOLO, Pandas, Altair, etc.).
   - Define constants (Font size, Confidence threshold, VOC classes).
2. Page Setup:
   - Configure Streamlit page layout and title.
3. Define Model Loading Logic:
   - Create a cached function 'load_models' to load YOLO models from disk.
   - Handle cases where models might be missing.
4. Define Helper Functions:
   - 'process_frame': Runs YOLO detection on an image frame and draws bounding boxes.
   - 'create_chart': Generates an Altair bar chart for detection counts.
   - 'fmt_pct': Formats float values as percentages.
   - 'get_eval_metric': Safely retrieves metrics from dictionaries.
5. Main UI Layout (Tabs):
   - Create two main tabs: "Live Demo" and "Research Metrics".
6. Tab 1: Live Demo Logic:
   - Initialize Session State variables for tracking detection counts.
   - Display Model Availability check.
   - 6.1. Controls:
         - Radio button for comparison mode (Baseline vs Naive OR Continual).
         - File uploader for input video.
   - 6.2. Video Processing Loop:
         - Save uploaded video to temp file.
         - Display Start/Stop buttons.
         - On 'Start', Run OpenCV loop to process video frame-by-frame.
         - Run models on frames, update counts, and refresh charts in real-time.
7. Tab 2: Research Metrics Logic:
   - 7.1. Load Data:
         - Read 'evaluation_results/metrics.json'.
   - 7.2. Compute/Display Tables:
         - Generate "Stability Analysis" table (Task 1 Performance).
         - Generate "Plasticity Analysis" table (Task 2 Performance).
         - Calculate "Forgetting Score" dynamically (Baseline - Current).
   - 7.3. Display Conclusions:
         - Analyze Stability (Did it forget?).
         - Analyze Plasticity (Did it learn new task?).
         - Display success/warning messages based on thresholds.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import cv2
import numpy as np
import tempfile
import pandas as pd
import altair as alt
import json


# 1. Imports and Configuration
FONT_SIZE = 24
YOLO_CONFIDENCE_THRESHOLD = 0.25

# PASCAL VOC Classes (20 classes)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Task mapping for coloring
VOC_TASKS = ['Task 1 (0-9)'] * 10 + ['Task 2 (10-19)'] * 10

# 2. Page Setup
st.set_page_config(page_title="Continual Learning Demo", layout="wide")
st.title("Continual Learning Dashboard")

# 3. Define Model Loading Logic
@st.cache_resource
def load_models():
    # Force cache reload
    models = {}
    path_t1 = "models/model_t1.pt"
    if os.path.exists(path_t1): models['Baseline'] = YOLO(path_t1)
    
    path_naive = "models/model_t2_naive.pt"
    if os.path.exists(path_naive): models['Naive'] = YOLO(path_naive)
        
    path_cont = "models/model_t2_continual.pt"
    if os.path.exists(path_cont): models['Continual'] = YOLO(path_cont)
        
    return models

loaded_models = load_models()
font = ImageFont.load_default()

# 4. Define Helper Functions

# 4.1. Process Frame (Detection & Drawing)
def process_frame(image_cv, model):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    results = model(image_pil, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    
    counts = {cls: 0 for cls in VOC_CLASSES}
    
    for box in results[0].boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        
        if cls_name in counts:
            counts[cls_name] += 1
            
        label = f"{cls_name} {conf:.2f}"
        box_color = "green"
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        draw.text((x1, y1 - 10), label, fill="white")

    return np.array(image_pil), counts

# 4.2. Create Altair Chart
def create_chart(counts, title):
    data = pd.DataFrame({
        'Class': VOC_CLASSES,
        'Count': [counts[c] for c in VOC_CLASSES],
        'Task': VOC_TASKS
    })
    
    c = alt.Chart(data).mark_bar().encode(
        x=alt.X('Class', sort=None, axis=alt.Axis(labelAngle=-45, title='Class (0-19)')),
        y=alt.Y('Count', title='Total Detections'),
        color=alt.Color('Task', scale=alt.Scale(domain=['Task 1 (0-9)', 'Task 2 (10-19)'], range=['#1f77b4', '#ff7f0e'])),
        tooltip=['Class', 'Count', 'Task']
    ).properties(
        title=title,
        height=300
    )
    return c

# 5. Main UI Layout (Tabs)
tab1, tab2 = st.tabs(["Live Demo", "Research Metrics"])

# 6. Tab 1: Live Demo Logic
with tab1:
    # 6.1. Initialize Session State
    if 'counts_baseline' not in st.session_state:
        st.session_state.counts_baseline = {cls: 0 for cls in VOC_CLASSES}
    if 'counts_naive' not in st.session_state:
        st.session_state.counts_naive = {cls: 0 for cls in VOC_CLASSES}
    if 'counts_continual' not in st.session_state:
        st.session_state.counts_continual = {cls: 0 for cls in VOC_CLASSES}
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False

    if not loaded_models:
        st.error("No models found! Please run training scripts first.")
    else:
        # 6.2. Controls
        comparison_mode = st.radio(
            "Select Comparison Mode:",
            ("Baseline vs 2nd training", "Continual Learning (Final)")
        )

        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="upload_demo")

        # Reset counts if new video is uploaded
        if uploaded_video:
            if 'last_video_name' not in st.session_state or st.session_state.last_video_name != uploaded_video.name:
                st.session_state.counts_baseline = {cls: 0 for cls in VOC_CLASSES}
                st.session_state.counts_naive = {cls: 0 for cls in VOC_CLASSES}
                st.session_state.counts_continual = {cls: 0 for cls in VOC_CLASSES}
                st.session_state.last_video_name = uploaded_video.name
                st.session_state.video_processed = False

        # 6.3. Baseline vs Naive Logic
        if uploaded_video is not None and comparison_mode == "Baseline vs 2nd training":
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            # Button Controls
            col_ctrl1, col_ctrl2 = st.columns(2)
            start_btn = col_ctrl1.button("Start / Resume Detection ▶️", key="start_base")
            stop_btn = col_ctrl2.button("Stop Detection ⏸️", key="stop_base")
            
            if stop_btn:
                st.session_state.video_processed = False
            
            if start_btn:
                st.session_state.video_processed = True
                st.session_state.counts_baseline = {cls: 0 for cls in VOC_CLASSES}
                st.session_state.counts_naive = {cls: 0 for cls in VOC_CLASSES}

            # Layout: Vertical Sections
            st.markdown("---")
            
            # Section 1: Baseline
            st.header("Baseline Model (Task 1 Only)")
            st.write("Trained on Classes 0-9. Should detect Vehicles/Animals, but miss Persons.")
            
            frame_ph_1 = st.empty()
            chart_ph_1 = st.empty()
            
            st.markdown("---")
            
            # Section 2: Naive
            st.header("Naive Model (Task 2 Only)")
            st.write("Trained on Classes 10-19. Should detect Persons, but FORGET Vehicles.")
            frame_ph_2 = st.empty()
            chart_ph_2 = st.empty()

            # Processing Loop
            if st.session_state.video_processed:
                cap = cv2.VideoCapture(tfile.name)
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 3rd frame
                    if frame_count % 3 == 0:
                        
                        # --- Baseline ---
                        if 'Baseline' in loaded_models:
                            img1, counts1 = process_frame(frame, loaded_models['Baseline'])
                            frame_ph_1.image(img1, channels="RGB", use_container_width=True)
                            
                            # Accumulate
                            for cls, c in counts1.items():
                                st.session_state.counts_baseline[cls] += c
                            
                            # Chart
                            chart1 = create_chart(st.session_state.counts_baseline, "Baseline Cumulative Counts")
                            chart_ph_1.altair_chart(chart1, use_container_width=True)
                        
                        # --- Naive ---
                        if 'Naive' in loaded_models:
                            img2, counts2 = process_frame(frame, loaded_models['Naive'])
                            frame_ph_2.image(img2, channels="RGB", use_container_width=True)
                            
                            # Accumulate
                            for cls, c in counts2.items():
                                st.session_state.counts_naive[cls] += c
                                
                            # Chart
                            chart2 = create_chart(st.session_state.counts_naive, "Naive Cumulative Counts")
                            chart_ph_2.altair_chart(chart2, use_container_width=True)
                    
                    frame_count += 1
                
                cap.release()
                st.success("Video Finished!")
                st.session_state.video_processed = False
                
            os.remove(tfile.name)

        # 6.4. Continual Logic
        elif uploaded_video is not None and comparison_mode == "Continual Learning (Final)":
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            # Button Controls
            col_ctrl1, col_ctrl2 = st.columns(2)
            start_btn = col_ctrl1.button("Start / Resume Detection ▶️", key="start_cont")
            stop_btn = col_ctrl2.button("Stop Detection ⏸️", key="stop_cont")
            
            if stop_btn:
                st.session_state.video_processed = False
            
            if start_btn:
                st.session_state.video_processed = True
                st.session_state.counts_continual = {cls: 0 for cls in VOC_CLASSES}

            # Layout
            st.markdown("---")
            
            # Section: Continual
            st.header("Continual Learning Model (Experience Replay)")
            st.write("Trained on Task 2 + Task 1 Exemplars. Should detect BOTH Persons AND Vehicles.")
            
            frame_ph_cont = st.empty()
            chart_ph_cont = st.empty()

            # Processing Loop
            if st.session_state.video_processed:
                cap = cv2.VideoCapture(tfile.name)
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 3rd frame
                    if frame_count % 3 == 0:
                        
                        # --- Continual ---
                        if 'Continual' in loaded_models:
                            img_cont, counts_cont = process_frame(frame, loaded_models['Continual'])
                            frame_ph_cont.image(img_cont, channels="RGB", use_container_width=True)
                            
                            # Accumulate
                            for cls, c in counts_cont.items():
                                st.session_state.counts_continual[cls] += c
                                
                            # Chart
                            chart_cont = create_chart(st.session_state.counts_continual, "Continual Model Cumulative Counts")
                            chart_ph_cont.altair_chart(chart_cont, use_container_width=True)
                    
                    frame_count += 1
                
                cap.release()
                st.success("Video Finished!")
                st.session_state.video_processed = False
                
            os.remove(tfile.name)

# 7. Tab 2: Research Metrics Logic
with tab2:
    st.header("Model Performance Analysis")
    
    # 7.1. Load Data
    metrics_file = "evaluation_results/metrics.json"
    if not os.path.exists(metrics_file):
        st.error(f"Metrics file not found at {metrics_file}. Please run 'python src/06_evaluate.py' first.")
    else:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            
        # Helper formatters
        def fmt_pct(val):
            if val is None: return "N/A"
            return f"{val * 100:.2f}%"

        def get_eval_metric(metrics_dict, key):
            if not metrics_dict: return None
            return metrics_dict.get(key, 0.0)

        # 7.2. Task 1 Performance (Stability)
        t1_rows = []
        models = ['Baseline', 'Naive', 'Continual']
        model_labels = {
            'Baseline': 'Baseline (Model T1)',
            'Naive': '2nd training without exemplar (Model T2)',
            'Continual': 'Continual (Model T2)'
        }
        
        # Get Baseline mAP@50 for dynamic calculation
        baseline_data = metrics_data.get('Baseline', {}).get('T1_Eval')
        baseline_map50 = None
        if baseline_data and 'map50' in baseline_data:
             baseline_map50 = baseline_data['map50']

        for m in models:
            data = metrics_data.get(m, {})
            t1_eval = data.get('T1_Eval')
            
            # Get current map50
            current_map50 = None
            if t1_eval and 'map50' in t1_eval:
                current_map50 = t1_eval['map50']
            
            # Calculate Forgetting Dynamically: Baseline - Current
            forgetting_val = None
            if m != 'Baseline' and baseline_map50 is not None and current_map50 is not None:
                forgetting_val = baseline_map50 - current_map50
            
            row = {
                'Model': model_labels[m],
                'Task 1 mAP@50': fmt_pct(current_map50),
                'Task 1 mAP@50-95': fmt_pct(get_eval_metric(t1_eval, 'map50-95')) if t1_eval else "N/A",
                'Task 1 Precision': fmt_pct(get_eval_metric(t1_eval, 'precision')) if t1_eval else "N/A",
                'Task 1 Recall': fmt_pct(get_eval_metric(t1_eval, 'recall')) if t1_eval else "N/A",
                'Forgetting mAP@50': fmt_pct(forgetting_val) if m != 'Baseline' else "N/A"
            }
            t1_rows.append(row)
            
        st.subheader("1. Stability Analysis (Task 1 Classes)")
        st.table(pd.DataFrame(t1_rows))

        # 7.2. Task 2 Performance (Plasticity)
        t2_rows = []
        for m in models:
            data = metrics_data.get(m, {})
            t2_eval = data.get('T2_Eval')
            
            # Baseline T2 is usually None or 0.
            if m == 'Baseline' and not t2_eval:
                row = {
                    'Model': model_labels[m],
                    'Task 2 mAP@50': "N/A",
                    'Task 2 mAP@50-95': "N/A",
                    'Task 2 Precision': "N/A",
                    'Task 2 Recall': "N/A"
                }
            else:
                 row = {
                    'Model': model_labels[m],
                    'Task 2 mAP@50': fmt_pct(get_eval_metric(t2_eval, 'map50')),
                    'Task 2 mAP@50-95': fmt_pct(get_eval_metric(t2_eval, 'map50-95')),
                    'Task 2 Precision': fmt_pct(get_eval_metric(t2_eval, 'precision')),
                    'Task 2 Recall': fmt_pct(get_eval_metric(t2_eval, 'recall'))
                }
            t2_rows.append(row)

        st.subheader("2. Plasticity Analysis (Task 2 Classes)")
        st.table(pd.DataFrame(t2_rows))
        
        # 7.3. Conclusions
        st.subheader("3. Conclusions")
        
        # Stability Conclusion (Forgetting)
        naive_forget = metrics_data.get('Naive', {}).get('Forgetting', 0.0)
        cont_forget = metrics_data.get('Continual', {}).get('Forgetting', 0.0)
        
        if isinstance(naive_forget, (int, float)) and isinstance(cont_forget, (int, float)):
             if cont_forget < naive_forget:
                improvement = naive_forget - cont_forget
                st.success(f"**Stability (Memory):** Experience Replay reduced forgetting by **{improvement:.4f}** mAP points compared to Direct Training.")
             else:
                st.warning("**Stability (Memory):** No significant reduction in forgetting observed.")
        
        # Plasticity Conclusion (New Learning)
        naive_t2 = get_eval_metric(metrics_data.get('Naive', {}).get('T2_Eval'), 'map50')
        cont_t2 = get_eval_metric(metrics_data.get('Continual', {}).get('T2_Eval'), 'map50')
        
        if isinstance(naive_t2, (int, float)) and isinstance(cont_t2, (int, float)):
            gap = naive_t2 - cont_t2
            # Interpret the gap
            if gap < 0.10: # Less than 10% drop is usually considered good for difficult CL
                st.success(f"**Plasticity (New Learning):** Strong Learning Capability. The Continual model achieved **{cont_t2:.2%}** mAP, only **{gap:.4f}** points lower than the dedicated model without examplar.")
            else:
                st.info(f"**Plasticity (New Learning):** Moderate Learning Capability. The model learned the new task but with a trade-off (**{gap:.4f}** points gap).")