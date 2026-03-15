import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from dataset_utils import PoisonedDataset, BackdoorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_autoencoder(dataset_path, name):
    print(f"Running Autoencoder Detector on {name}...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    data = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        data.append(img.view(-1).numpy())
    
    data = np.array(data)
    tensor_data = torch.from_numpy(data).float()
    
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for a few epochs - in a real scenario, this would be trained on clean data
    # Here we demonstrate the concept by training on the (possibly poisoned) dataset
    # and identifying samples with highest reconstruction error.
    dataloader = DataLoader(tensor_data, batch_size=64, shuffle=True)
    
    print("Training autoencoder...")
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    model.eval()
    with torch.no_grad():
        reconstructions = model(tensor_data)
        # Compute per-sample MSE
        errors = torch.mean((reconstructions - tensor_data)**2, dim=1).numpy()
    
    # Flag samples with error above 90th percentile
    threshold = np.percentile(errors, 90)
    anomaly_flags = (errors > threshold).astype(int)
    
    output_path = f"data/processed/{name.lower().replace(' ', '_')}_ae_flags.npy"
    np.save(output_path, anomaly_flags)
    
    # Evaluate against ground truth
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
        
    run_autoencoder("data/processed/label_flip_dataset.pkl", "Label Flip")
    run_autoencoder("data/processed/backdoor_dataset.pkl", "Backdoor")

if __name__ == "__main__":
    main()
