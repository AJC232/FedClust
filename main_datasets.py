import os
import torchvision
import torchvision.transforms as transforms
from arguments import getArgs

# Get the arguments
args = getArgs()

# Create the datasets directory if it does not exist
os.makedirs(args.drive_path + 'datasets', exist_ok=True)

# Define transformations for CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize with mean and std of CIFAR-10
    transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.20220212, 0.19931542, 0.20086347))  # Normalize with mean and std of CIFAR-10
])

# CIFAR-10 Dataset
cifar10_trainset = torchvision.datasets.CIFAR10(root=args.drive_path + 'datasets/cifar10_data', train=True, transform=transform, download=True)
cifar10_testset = torchvision.datasets.CIFAR10(root=args.drive_path + 'datasets/cifar10_data', train=False, transform=transform, download=True)

# Define transformations for SVHN dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052)),
])

# SVHN Dataset
svhn_trainset = torchvision.datasets.SVHN(root=args.drive_path + 'datasets/svhn_data', split='train', transform=transform, download=True)
svhn_testset = torchvision.datasets.SVHN(root=args.drive_path + 'datasets/svhn_data', split='test', transform=transform, download=True)

# Define transformations for MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Replicate the single channel to 3 channels
])

# MNIST Dataset
mnist_trainset = torchvision.datasets.MNIST(root=args.drive_path + 'datasets/mnist_data', train=True, transform=transform, download=True)
mnist_testset = torchvision.datasets.MNIST(root=args.drive_path + 'datasets/mnist_data', train=False, transform=transform, download=True)

# Define transformations for Fashion MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3205,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Replicate the single channel to 3 channels
])

# Fashion MNIST Dataset
fmnist_trainset = torchvision.datasets.FashionMNIST(root=args.drive_path + 'datasets/fmnist_data', train=True, transform=transform, download=True)
fmnist_testset = torchvision.datasets.FashionMNIST(root=args.drive_path + 'datasets/fmnist_data', train=False, transform=transform, download=True)

# Function to load CIFAR-10 dataset
def load_cifar10_dataset(train):
  if train == True:
    return cifar10_trainset
  else:
    return cifar10_testset

# Function to load MNIST dataset
def load_mnist_dataset(train):
  if train == True:
    return mnist_trainset
  else:
    return mnist_testset

# Function to load SVHN dataset
def load_svhn_dataset(train):
  if train == True:
    return svhn_trainset
  else:
    return svhn_testset

# Function to load Fashion MNIST dataset
def load_fmnist_dataset(train):
  if train == True:
    return fmnist_trainset
  else:
    return fmnist_testset