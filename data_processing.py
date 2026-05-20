import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

BATCH_SIZE = 128

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

full_train_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True
)

train_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=train_transform
)

val_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=val_transform
)

train_size = int(0.9 * len(full_train_data))
val_size = len(full_train_data) - train_size

indices = torch.randperm(len(full_train_data), generator=torch.Generator().manual_seed(42))

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_set = Subset(train_data, train_indices)
val_set = Subset(val_data, val_indices)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(len(train_loader) * 128)
