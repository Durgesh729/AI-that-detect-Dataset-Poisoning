from torchvision import datasets, transforms

transform = transforms.ToTensor()

cifar_dataset = datasets.CIFAR10(
    root="../data/raw",
    train=True,
    download=True,
    transform=transform
)

print("CIFAR10 downloaded successfully")
print("Total samples:", len(cifar_dataset))