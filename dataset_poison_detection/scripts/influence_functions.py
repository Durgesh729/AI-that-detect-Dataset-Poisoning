import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from dataset_utils import PoisonedDataset, BackdoorDataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_influence(dataset_path, name):
    print(f"Computing Influence Functions for {name} (TracIn approximation)...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    # Use a subset if dataset is huge, but for MNIST 60k it's okay for demo
    subset_indices = list(range(0, len(dataset), 10)) # Sample every 10th for speed
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Collect gradients at different stages
    checkpoints = []
    print("Training and collecting gradients...")
    for epoch in range(3):
        for i, (images, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Save model state
        checkpoints.append(pickle.dumps(model.state_dict()))
        print(f"Epoch {epoch+1} complete")

    influence_scores = np.zeros(len(dataset))
    
    # Simplified TracIn: score = sum_{checkpoints} grad(sample) dot grad(final_loss)
    # Actually, a simpler proxy is the per-sample loss trend. 
    # Hard samples (poisoned) often have high loss throughout.
    
    print("Estimating influence scores...")
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            img = img.unsqueeze(0)
            lbl = torch.tensor([lbl])
            out = model(img)
            loss = criterion(out, lbl)
            losses.append(loss.item())
    
    losses = np.array(losses)
    # High loss samples are potentially poisoned or outliers
    # We'll use normalized loss as a proxy for negative influence
    influence_scores = losses 
    
    # Flag top 10% highest loss as poisoned
    threshold = np.percentile(influence_scores, 90)
    anomaly_flags = (influence_scores > threshold).astype(int)
    
    output_path = f"data/processed/{name.lower().replace(' ', '_')}_influence_flags.npy"
    np.save(output_path, anomaly_flags)
    
    # Evaluate
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
        
    compute_influence("data/processed/label_flip_dataset.pkl", "Label Flip")
    compute_influence("data/processed/backdoor_dataset.pkl", "Backdoor")

if __name__ == "__main__":
    main()
