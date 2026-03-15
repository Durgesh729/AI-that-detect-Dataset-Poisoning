import torch
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poison_indices, target_labels):
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.target_labels = target_labels # Dictionary mapping index to new label

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if index in self.poison_indices:
            label = self.target_labels[index]
        return img, label

    def __len__(self):
        return len(self.base_dataset)

def add_backdoor_trigger(img):
    """Adds a small 3x3 white square in the bottom right corner."""
    img_copy = img.clone()
    img_copy[0, 24:27, 24:27] = 1.0
    return img_copy

class BackdoorDataset(Dataset):
    def __init__(self, base_dataset, poison_indices, target_label):
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.target_label = target_label

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if index in self.poison_indices:
            img = add_backdoor_trigger(img)
            label = self.target_label
        return img, label

    def __len__(self):
        return len(self.base_dataset)
