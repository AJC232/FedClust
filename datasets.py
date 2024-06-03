import torchvision
import torchvision.transforms as transforms

# Define transformations for the training and test sets
cifar10_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize with mean and std of CIFAR-10
    transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.20220212, 0.19931542, 0.20086347))  # Normalize with mean and std of CIFAR-10
])

def load_cifar10_dataset(train):
  if train == True:
    cifar10_trainset = torchvision.datasets.CIFAR10(root='cifar10_data', train=True, transform=cifar10_transform, download=True)
    return cifar10_trainset
  else:
    cifar10_testset = torchvision.datasets.CIFAR10(root='cifar10_data', train=False, transform=cifar10_transform, download=True)
    return cifar10_testset
  
# MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Replicate the single channel to 3 channels
])

def load_mnist_dataset(train):
  if train == True:
    mnist_trainset = torchvision.datasets.MNIST(root='mnist_data', train=True, transform=mnist_transform, download=True)
    return mnist_trainset
  else:
    mnist_testset = torchvision.datasets.MNIST(root='mnist_data', train=False, transform=mnist_transform, download=True)
    return mnist_testset
  
# USPS
usps_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2469), (0.2811)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Replicate the single channel to 3 channels
])

def load_usps_dataset(train):
  if train == True:
    usps_trainset = torchvision.datasets.USPS(root='usps_data', train=True, transform=usps_transform, download=True)
    return usps_trainset
  else:
    usps_testset = torchvision.datasets.USPS(root='usps_data', train=False, transform=usps_transform, download=True)
    return usps_testset
  
# SVHN
svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052)),
])


def load_svhn_dataset(train):
  if train == True:
    svhn_trainset = torchvision.datasets.SVHN(root='svhn_data', split='train', transform=svhn_transform, download=True)
    return svhn_trainset
  else:
    svhn_testset = torchvision.datasets.SVHN(root='svhn_data', split='test', transform=svhn_transform, download=True)
    return svhn_testset