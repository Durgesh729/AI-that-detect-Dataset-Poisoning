---
description: How to run and extend the Dataset Poisoning Detection Pipeline
---

This workflow guides you through executing the full dataset poisoning detection and defense pipeline.

### 1. Environment Setup
Ensure all dependencies are installed using the `requirements.txt` file in the root.
// turbo
```powershell
pip install -r requirements.txt
```

### 2. Data Initialization
Download the base datasets (MNIST/CIFAR) before running the pipeline.
// turbo
```powershell
python dataset_poison_detection/scripts/download_mnist.py
python dataset_poison_detection/scripts/download_cifar.py
```

### 3. Running the Full Pipeline
Navigate to the project directory and run the orchestrator. This will sequentially perform:
- Poison Attack Simulation
- Feature Profiling (Visualizations)
- Multiple Detection Runs (Isolation Forest, AE, Influence, Trust)
- Final Training & Defense Evaluation

```powershell
cd dataset_poison_detection
python main.py
```

### 4. Viewing Results
- **Visual Evidence**: Check `data/processed/` for `.png` scatter plots (t-SNE/PCA).
- **Detection Metrics**: Review terminal output for Precision and Recall per stage.
- **Interactive Exploration**: Open `notebooks/exploration.ipynb` in VS Code to see specific flagged images.
- **Final Report**: Read `data/processed/evaluation_results.txt` for the final Accuracy and Attack Success Rate (ASR) comparison.

### 5. Extending the Pipeline
To add a new detection method:
1. Create a new script in `scripts/`.
2. Follow the pattern in `scripts/isolation_forest_detector.py` (load pkl, detect, save npy).
3. Add the script name to the `pipeline_stages` list in `main.py`.
