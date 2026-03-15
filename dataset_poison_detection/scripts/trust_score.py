import torch
import torch.nn as nn
import pickle
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from dataset_utils import PoisonedDataset, BackdoorDataset

def compute_trust_score(dataset_path, name):
    print(f"Computing Trust Scores for {name}...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    data = []
    labels = []
    # Use a subset for kNN speed
    subset_size = min(5000, len(dataset))
    for i in range(subset_size):
        img, lbl = dataset[i]
        data.append(img.view(-1).numpy())
        labels.append(lbl)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Trust Score Concept:
    # 1. Distance to the nearest class (d_yc)
    # 2. Distance to the nearest other class (d_not_yc)
    # Trust = d_not_yc / d_yc (simplified)
    
    print("Fitting kNN...")
    # Find nearest neighbors across all points
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(data)
    
    trust_scores = []
    for i in range(subset_size):
        # Samples of same label
        same_class_mask = (labels == labels[i])
        other_class_mask = (labels != labels[i])
        
        # This is a bit slow for large kNN, simplified version:
        # Just check if majority of 10-NN have the same label
        indices = nbrs.kneighbors([data[i]], return_distance=False)[0]
        neighbor_labels = labels[indices]
        
        # Trust = proportion of neighbors with same label
        trust = np.mean(neighbor_labels == labels[i])
        trust_scores.append(trust)
    
    trust_scores = np.array(trust_scores)
    # Low trust -> suspect poisoned
    threshold = np.percentile(trust_scores, 10)
    anomaly_flags = (trust_scores <= threshold).astype(int)
    
    output_path = f"data/processed/{name.lower().replace(' ', '_')}_trust_flags.npy"
    np.save(output_path, anomaly_flags)
    
    # Evaluate
    gt_path = "data/processed/poison_indices.npy"
    if os.path.exists(gt_path):
        poison_indices = np.load(gt_path)
        # Filter GT to subset size
        gt_flags = np.zeros(subset_size)
        for idx in poison_indices:
            if idx < subset_size:
                gt_flags[idx] = 1
        
        tp = np.sum((anomaly_flags == 1) & (gt_flags == 1))
        fp = np.sum((anomaly_flags == 1) & (gt_flags == 0))
        fn = np.sum((anomaly_flags == 0) & (gt_flags == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Results for {name} (Subset): Precision: {precision:.4f}, Recall: {recall:.4f}")

def main():
    if not os.path.exists("data/processed/label_flip_dataset.pkl"):
        print("Please run poison_attack.py first.")
        return
        
    compute_trust_score("data/processed/label_flip_dataset.pkl", "Label Flip")
    compute_trust_score("data/processed/backdoor_dataset.pkl", "Backdoor")

if __name__ == "__main__":
    main()
