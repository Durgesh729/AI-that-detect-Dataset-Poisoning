import torch
import numpy as np
import random
import pickle
import os
from torchvision import datasets, transforms
from dataset_utils import PoisonedDataset, BackdoorDataset

def main():
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    
    total_samples = len(mnist_train)
    poison_rate = 0.1
    num_poison = int(total_samples * poison_rate)
    
    all_indices = list(range(total_samples))
    poison_indices = random.sample(all_indices, num_poison)
    
    # 1. Label Flip Attack
    target_labels = {idx: (mnist_train.targets[idx].item() + 1) % 10 for idx in poison_indices}
    label_flip_dataset = PoisonedDataset(mnist_train, poison_indices, target_labels)
    
    # 2. Backdoor Attack
    # Target all poisoned samples to label '0'
    backdoor_dataset = BackdoorDataset(mnist_train, poison_indices, 0)
    
    # Save Metadata
    np.save("data/processed/poison_indices.npy", np.array(poison_indices))
    
    # Save Datasets
    with open("data/processed/label_flip_dataset.pkl", "wb") as f:
        pickle.dump(label_flip_dataset, f)
    
    with open("data/processed/backdoor_dataset.pkl", "wb") as f:
        pickle.dump(backdoor_dataset, f)
        
    print(f"Attack Simulation Complete.")
    print(f"Poisoned {num_poison} samples ({poison_rate*100}%).")
    print(f"Saved datasets and ground truth indices to data/processed/")

if __name__ == "__main__":
    main()
