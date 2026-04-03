import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import sys

# Add script directory to path so we can import dataset_utils if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import PoisonedDataset, BackdoorDataset

def load_data():
    # The script is in dataset_poison_detection/scripts/
    # Data is in dataset_poison_detection/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(base_dir, "data", "processed")
    raw_dir = os.path.join(base_dir, "data", "raw")
    
    # Load MNIST
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root=raw_dir, train=True, download=True, transform=transform)
    
    # Load Ground Truth
    poison_indices = np.load(os.path.join(processed_dir, "poison_indices.npy"))
    gt_flags = np.zeros(len(mnist_train))
    gt_flags[poison_indices] = 1
    
    # Load Datasets
    with open(os.path.join(processed_dir, "backdoor_dataset.pkl"), "rb") as f:
        backdoor_dataset = pickle.load(f)
        
    # Load Detecion Flags (Isolation Forest as example)
    if_flags_path = os.path.join(processed_dir, "backdoor_if_flags.npy")
    if os.path.exists(if_flags_path):
        det_flags = np.load(if_flags_path)
    else:
        det_flags = np.zeros(len(mnist_train))
        
    return mnist_train, backdoor_dataset, gt_flags, det_flags, poison_indices

def create_report():
    print("Generating Visual Report...")
    mnist_train, backdoor_dataset, gt_flags, det_flags, poison_indices = load_data()
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4)
    
    # Row 1: Clean Samples
    clean_indices = [i for i in range(100) if i not in poison_indices][:5]
    for i, idx in enumerate(clean_indices):
        img, label = mnist_train[idx]
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].set_title(f"Clean (L:{label})")
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel("Clean Samples", size='large')
    
    # Row 2: Poisoned (Backdoor) Samples
    p_indices = poison_indices[:5]
    for i, idx in enumerate(p_indices):
        img, label = backdoor_dataset[idx]
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        axes[1, i].set_title(f"Poisoned (L:{label})")
        axes[1, i].axis('off')
    
    # Row 3: Detection Examples (TP, FP, FN)
    tp_indices = np.where((det_flags == 1) & (gt_flags == 1))[0][:2]
    fn_indices = np.where((det_flags == 0) & (gt_flags == 1))[0][:2]
    fp_indices = np.where((det_flags == 1) & (gt_flags == 0))[0][:1]
    
    row3_indices = list(tp_indices) + list(fn_indices) + list(fp_indices)
    row3_labels = ["TP", "TP", "FN", "FN", "FP"]
    
    for i, idx in enumerate(row3_indices[:5]):
        img, label = backdoor_dataset[idx] if gt_flags[idx] == 1 else mnist_train[idx]
        axes[2, i].imshow(img.squeeze(), cmap='gray')
        axes[2, i].set_title(f"{row3_labels[i]} (Idx:{idx})")
        axes[2, i].axis('off')

    # Add descriptive text
    plt.suptitle("Dataset Poisoning Detection Report - Backdoor Attack", fontsize=20)
    plt.figtext(0.5, 0.02, "TP: True Positive (Poison detected) | FN: False Negative (Poison missed) | FP: False Positive (Clean flagged)", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Fix: Define processed_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(base_dir, "data", "processed")

    output_path = os.path.join(processed_dir, "detection_visual_report.png")
    plt.savefig(output_path)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    create_report()
