import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from dataset_utils import PoisonedDataset, BackdoorDataset

# Simple CNN for evaluation
class EvaluationCNN(nn.Module):
    def __init__(self):
        super(EvaluationCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def train_model(train_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvaluationCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    print("Stage 8: Defense and Evaluation")
    print("===============================")
    
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="data/raw", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Define Backdoor Trigger for evaluation
    def apply_trigger(img):
        img_copy = img.clone()
        img_copy[0, 24:27, 24:27] = 1.0 # Same as in dataset_utils
        return img_copy

    # 1. Baseline (Clean)
    print("\n[1/3] Training Baseline Model (Clean Data)...")
    clean_train_set = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    clean_loader = DataLoader(clean_train_set, batch_size=64, shuffle=True)
    baseline_model = train_model(clean_loader, epochs=3)
    baseline_acc = evaluate(baseline_model, test_loader)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")

    # 2. Poisoned (Backdoor)
    print("\n[2/3] Training Poisoned Model (Backdoor Attack)...")
    with open("data/processed/backdoor_dataset.pkl", "rb") as f:
        backdoor_train_set = pickle.load(f)
    backdoor_loader = DataLoader(backdoor_train_set, batch_size=64, shuffle=True)
    poisoned_model = train_model(backdoor_loader, epochs=3)
    poisoned_acc = evaluate(poisoned_model, test_loader)
    
    # Calculate Backdoor Success Rate (ASR)
    # Target label was 0 in poison_attack.py
    triggered_test_data = []
    for i in range(len(test_set)):
        img, _ = test_set[i]
        triggered_test_data.append((apply_trigger(img), 0))
    
    triggered_loader = DataLoader(triggered_test_data, batch_size=64, shuffle=False)
    asr = evaluate(poisoned_model, triggered_loader)
    
    print(f"Poisoned Model Accuracy (Clean Test): {poisoned_acc:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")

    # 3. Defended (Filtered using Ensemble of ALL Detection Flags)
    print("\n[3/3] Training Defended Model (Filtered via Ensemble Union)...")
    
    # Load all available flags
    ae_flags = np.load("data/processed/backdoor_ae_flags.npy").astype(bool)
    if_flags = np.load("data/processed/backdoor_if_flags.npy").astype(bool)
    inf_flags = np.load("data/processed/backdoor_influence_flags.npy").astype(bool)
    
    # Trust flags might be subsetted, handle padding
    trust_flags_raw = np.load("data/processed/backdoor_trust_flags.npy")
    trust_flags = np.zeros(len(ae_flags), dtype=bool)
    trust_flags[:len(trust_flags_raw)] = trust_flags_raw.astype(bool)
    
    # ENSEMBLE: Union of all detections
    ensemble_flags = ae_flags | if_flags | inf_flags | trust_flags
    
    # Identify indices NOT flagged as anomalous
    safe_indices = np.where(ensemble_flags == False)[0]
    
    # Create Filtered Subset
    defended_train_set = Subset(backdoor_train_set, safe_indices)
    print(f"Ensemble filtered out {np.sum(ensemble_flags)} samples. Training on {len(safe_indices)} remaining.")
    
    defended_loader = DataLoader(defended_train_set, batch_size=64, shuffle=True)
    defended_model = train_model(defended_loader, epochs=3)
    defended_acc = evaluate(defended_model, test_loader)
    defended_asr = evaluate(defended_model, triggered_loader)
    
    print(f"Defended Model Accuracy (Clean Test): {defended_acc:.2f}%")
    print(f"Defended Model ASR: {defended_asr:.2f}%")

    # 4. Defended (Label Flip)
    print("\n[4/4] Training Defended Model (Label Flip Data)...")
    with open("data/processed/label_flip_dataset.pkl", "rb") as f:
        lf_train_set = pickle.load(f)
    
    # Load Label Flip Flags
    lf_ae_flags = np.load("data/processed/label_flip_ae_flags.npy").astype(bool)
    lf_if_flags = np.load("data/processed/label_flip_if_flags.npy").astype(bool)
    lf_inf_flags = np.load("data/processed/label_flip_influence_flags.npy").astype(bool)
    
    lf_trust_flags_raw = np.load("data/processed/label_flip_trust_flags.npy")
    lf_trust_flags = np.zeros(len(lf_ae_flags), dtype=bool)
    lf_trust_flags[:len(lf_trust_flags_raw)] = lf_trust_flags_raw.astype(bool)
    
    lf_ensemble = lf_ae_flags | lf_if_flags | lf_inf_flags | lf_trust_flags
    lf_safe_indices = np.where(lf_ensemble == False)[0]
    
    lf_defended_set = Subset(lf_train_set, lf_safe_indices)
    lf_defended_loader = DataLoader(lf_defended_set, batch_size=64, shuffle=True)
    lf_defended_model = train_model(lf_defended_loader, epochs=3)
    lf_defended_acc = evaluate(lf_defended_model, test_loader)
    
    # Evaluate Clean Acc on Poisoned Label Flip (for comparison)
    lf_loader = DataLoader(lf_train_set, batch_size=64, shuffle=True)
    lf_poison_model = train_model(lf_loader, epochs=3)
    lf_poison_acc = evaluate(lf_poison_model, test_loader)

    print(f"Label Flip Poisoned Acc: {lf_poison_acc:.2f}%")
    print(f"Label Flip Defended Acc: {lf_defended_acc:.2f}%")
    
    print("\nFINAL PIPELINE EVALUATION SUMMARY:")
    print("-" * 50)
    print(f"{'Attack':<15} | {'Model':<10} | {'Clean Acc':<10} | {'ASR':<10}")
    print("-" * 50)
    print(f"{'Baseline':<15} | {'Clean':<10} | {baseline_acc:8.2f}% | {'N/A':<10}")
    print(f"{'Backdoor':<15} | {'Poisoned':<10} | {poisoned_acc:8.2f}% | {asr:8.2f}%")
    print(f"{'Backdoor':<15} | {'Defended':<10} | {defended_acc:8.2f}% | {defended_asr:8.2f}%")
    print(f"{'Label Flip':<15} | {'Poisoned':<10} | {lf_poison_acc:8.2f}% | {'N/A':<10}")
    print(f"{'Label Flip':<15} | {'Defended':<10} | {lf_defended_acc:8.2f}% | {'N/A':<10}")
    print("-" * 50)
    
    # Save results to a file
    with open("data/processed/evaluation_results.txt", "w") as f:
        f.write("Dataset Poisoning Evaluation Results\n")
        f.write("====================================\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
        f.write(f"Backdoor (Poisoned) Accuracy: {poisoned_acc:.2f}%\n")
        f.write(f"Backdoor (Poisoned) ASR: {asr:.2f}%\n")
        f.write(f"Backdoor (Defended) Accuracy: {defended_acc:.2f}%\n")
        f.write(f"Backdoor (Defended) ASR: {defended_asr:.2f}%\n")
        f.write(f"Label Flip (Poisoned) Accuracy: {lf_poison_acc:.2f}%\n")
        f.write(f"Label Flip (Defended) Accuracy: {lf_defended_acc:.2f}%\n")

if __name__ == "__main__":
    main()
