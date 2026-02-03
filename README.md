**Preventing Catastrophic Forgetting in Object Detection using Experience Replay.**

## Quick Start

### 1. Install
```bash
git clone https://github.com/MJTham/FinalYearProject.git
cd FinalYearProject
pip install -r requirements.txt
```

### 2. Run the Pipeline
Run the scripts in order to reproduce the experiment:
```bash
python src/01_prepare_data.py      # Download & Split Data
python src/02_train_baseline.py    # Train Task 1
python src/03_train_naive.py       # Train Task 2 (Forgetting)
python src/04_herding.py           # Build Memory (Exemplars)
python src/05_train_continual.py   # Train Task 2 + Memory
python src/06_evaluate.py          # Generate Report
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## Project Structure
```text
src/
├── 01_prepare_data.py      # Data Setup
├── 02_train_baseline.py    # Phase 1 Training
├── 03_train_naive.py       # Phase 2 Training (Naive)
├── 04_herding.py           # Memory Selection
├── 05_train_continual.py   # Phase 3 Training (Continual)
├── 06_evaluate.py          # Evaluation
└── utils/                  # Helper Libraries
```

## Results
The **Continual Model** successfully retains knowledge of Task 1 (Old Classes) while learning Task 2 (New Classes), significantly outperforming the Naive approach which suffers from Catastrophic Forgetting.

Check `evaluation_results/` for detailed charts and metrics.
