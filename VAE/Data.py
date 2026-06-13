import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

#Convert to tensor and normalise MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load MNIST data
mnist_train = datasets.MNIST(root='./_data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./_data', train=False, download=True, transform=transform)

#Remove labels and flatten images
mnist_train_images = torch.stack([data[0] for data in mnist_train])
mnist_test_images = torch.stack([data[0] for data in mnist_test])
mnist_train_images = mnist_train_images.view(mnist_train_images.size(0), -1)
mnist_test_images = mnist_test_images.view(mnist_test_images.size(0), -1)

#Move images to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_train_images = mnist_train_images.to(device)
mnist_test_images = mnist_test_images.to(device)

#Batch
train_dataset = TensorDataset(mnist_train_images)
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)