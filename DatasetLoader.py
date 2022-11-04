from torchvision.transforms import transforms
import torchvision
import torch

def get_dataset_loader(dataset_path, batch_size, input_size):
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32), 
        test_transform
    ])
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
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataset_loader('minist', batch_size=128)
    niter = iter(train_loader)
    image, label = next(niter)
    print(image.shape, label.shape)
