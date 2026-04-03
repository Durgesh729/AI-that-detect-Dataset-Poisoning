import torch
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
import os
from dataset_utils import PoisonedDataset, BackdoorDataset

def run_isolation_forest(dataset_path, name):
    print(f"Running Isolation Forest on {name}...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    data = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        data.append(img.view(-1).numpy())
    
    data = np.array(data)
    
    # Fit Isolation Forest with optimized parameters
    # Doubling contamination and estimators to improve recall
    clf = IsolationForest(contamination=0.2, n_estimators=200, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(data)
    
    # IsolationForest returns -1 for outliers and 1 for inliers
    anomaly_flags = (preds == -1).astype(int)
    
    output_path = f"data/processed/{name.lower().replace(' ', '_')}_if_flags.npy"
    np.save(output_path, anomaly_flags)
    print(f"Detection flags saved to {output_path}")
    
    # Evaluate if ground truth exists
    gt_path = "data/processed/poison_indices.npy"
    if os.path.exists(gt_path):
        poison_indices = np.load(gt_path)
        gt_flags = np.zeros(len(dataset))
        gt_flags[poison_indices] = 1
        
        tp = np.sum((anomaly_flags == 1) & (gt_flags == 1))
        fp = np.sum((anomaly_flags == 1) & (gt_flags == 0))
        fn = np.sum((anomaly_flags == 0) & (gt_flags == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Results for {name}: Precision: {precision:.4f}, Recall: {recall:.4f}")

def main():
    if not os.path.exists("data/processed/label_flip_dataset.pkl"):
        print("Please run poison_attack.py first.")
        return
        
    run_isolation_forest("data/processed/label_flip_dataset.pkl", "Label Flip")
    run_isolation_forest("data/processed/backdoor_dataset.pkl", "Backdoor")

if __name__ == "__main__":
    main()
