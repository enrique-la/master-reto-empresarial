"""pytorchexample: A Flower / PyTorch app."""

import datasets
datasets.disable_caching()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, PathologicalPartitioner


class Net(nn.Module):
    """CNN para FashionMNIST (28x28 escala de grises)."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)   # 1 canal (grayscale)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _get_partitioner(partitioner_type: str, num_partitions: int,
                     alpha: float = 0.5, num_classes: int = 2):
    if partitioner_type == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            min_partition_size=10,
            self_balancing=True,
        )
    elif partitioner_type == "pathological":
        return PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_classes_per_partition=num_classes,
        )
    else:  # iid (default)
        return IidPartitioner(num_partitions=num_partitions)


def load_data(partition_id: int, num_partitions: int, batch_size: int,
              partitioner_type: str = "iid", alpha: float = 0.5,
              num_classes_per_partition: int = 2):
    """Carga la partición del cliente con el particionador especificado."""
    partitioner = _get_partitioner(partitioner_type, num_partitions,
                                   alpha, num_classes_per_partition)
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": partitioner},
    )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_data = partition_train_test["train"].with_transform(apply_transforms)
    test_data  = partition_train_test["test"].with_transform(apply_transforms)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader   = DataLoader(test_data,  batch_size=batch_size)
    return trainloader, valloader


def load_centralized_dataset(batch_size: int = 32):
    """Carga el test set completo para evaluación global en el servidor."""
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": IidPartitioner(num_partitions=1)},
    )
    test_data = fds.load_split("test")

    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    test_data = test_data.with_transform(apply_transforms)
    return DataLoader(test_data, batch_size=batch_size)


def train(model, trainloader, epochs, lr, device):
    """Entrena el modelo una época local."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    total_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(trainloader)


def test(model, testloader, device):
    """Evalúa el modelo en el conjunto de test."""
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = total_loss / len(testloader)
    return loss, accuracy
