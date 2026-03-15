from torchvision import datasets, transforms

transform = transforms.ToTensor()

mnist_dataset = datasets.MNIST(
    root="../data/raw",
    train=True,
    download=True,
    transform=transform
)

print("MNIST downloaded successfully")
print("Total samples:", len(mnist_dataset))