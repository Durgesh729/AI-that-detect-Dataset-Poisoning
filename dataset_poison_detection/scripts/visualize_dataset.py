import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.ToTensor()

dataset = datasets.MNIST(root="../data/raw", train=True, download=True, transform=transform)

image, label = dataset[0]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.show()