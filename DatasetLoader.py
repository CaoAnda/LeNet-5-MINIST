from torchvision.transforms import transforms
import torchvision
import torch

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(32), 
    test_transform
])

def get_dataset_loader(dataset_path, batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_path,
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_path,
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader