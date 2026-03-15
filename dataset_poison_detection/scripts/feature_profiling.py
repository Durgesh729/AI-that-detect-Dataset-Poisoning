import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from dataset_utils import PoisonedDataset, BackdoorDataset

def profile_dataset(dataset_path, name):
    print(f"Profiling {name}...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    data = []
    labels = []
    # Take a subset for profiling speed if large
    subset_size = min(2000, len(dataset))
    for i in range(subset_size):
        img, lbl = dataset[i]
        data.append(img.view(-1).numpy())
        labels.append(lbl)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
    plt.title(f'PCA - {name}')
    plt.colorbar()
    
    # t-SNE
    print(f"Running t-SNE for {name} (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(data)
    
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
    plt.title(f't-SNE - {name}')
    plt.colorbar()
    
    output_path = f"data/processed/profile_{name.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved profile plot to {output_path}")
    plt.close()

def main():
    if not os.path.exists("data/processed/label_flip_dataset.pkl"):
        print("Please run poison_attack.py first.")
        return
        
    profile_dataset("data/processed/label_flip_dataset.pkl", "Label Flip Dataset")
    profile_dataset("data/processed/backdoor_dataset.pkl", "Backdoor Dataset")

if __name__ == "__main__":
    main()
