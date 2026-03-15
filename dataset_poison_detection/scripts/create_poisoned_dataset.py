import random
import pickle
import copy
from torchvision import datasets, transforms

transform = transforms.ToTensor()

# Load clean MNIST dataset
dataset = datasets.MNIST(root="../data/raw", train=True, download=True, transform=transform)

# Save clean dataset
with open("../data/clean_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)
print("Clean dataset saved to data/clean_dataset.pkl")
print("Total clean samples:", len(dataset))

# ──────────────────────────────────────────────
def poison_labels(dataset, poison_rate=0.1):
    """Randomly flip labels for a fraction of samples."""
    poisoned = copy.deepcopy(dataset)
    total = len(poisoned)
    num_poison = int(total * poison_rate)

    poisoned_indices = random.sample(range(total), num_poison)
    for i in poisoned_indices:
        poisoned.targets[i] = (poisoned.targets[i] + 1) % 10

    print(f"Poisoned {num_poison} out of {total} samples ({poison_rate*100:.0f}%)")
    return poisoned

# Create poisoned dataset
poisoned_dataset = poison_labels(dataset, poison_rate=0.1)

# Save poisoned dataset
with open("../data/poisoned_dataset.pkl", "wb") as f:
    pickle.dump(poisoned_dataset, f)
print("Poisoned dataset saved to data/poisoned_dataset.pkl")