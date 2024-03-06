import os
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import transforms

from ibda.utils import save_as_np


def subset_selection(dataset, labels, ratio, random_seed):
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    else:
        labels = np.array(labels)
    if ratio == 1:
        return dataset, labels, np.arange(len(labels))
    else:
        (
            sel_ids,
            _,
        ) = train_test_split(
            np.arange(len(labels)),
            train_size=ratio,
            random_state=random_seed,
            stratify=labels,
        )
        return Subset(dataset, sel_ids), labels[sel_ids], sel_ids


def load_mnist(data_folder_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    trainset = MNIST(
        root=data_folder_path, download=True, train=True, transform=transform
    )
    testset = MNIST(
        root=data_folder_path, download=True, train=False, transform=transform
    )
    return trainset, testset, trainset.targets, testset.targets


def load_fmnist(data_folder_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    trainset = FashionMNIST(
        root=data_folder_path, download=True, train=True, transform=transform
    )
    testset = FashionMNIST(
        root=data_folder_path, download=True, train=False, transform=transform
    )
    return trainset, testset, trainset.targets, testset.targets


def load_cifar10(data_folder_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    trainset = CIFAR10(
        root=data_folder_path, download=True, train=True, transform=transform
    )
    trainset.targets = torch.tensor(trainset.targets)
    testset = CIFAR10(
        root=data_folder_path, download=True, train=False, transform=transform
    )
    testset.targets = torch.tensor(testset.targets)
    return trainset, testset, trainset.targets, testset.targets


dispatcher = {
    "mnist": load_mnist,
    "fmnist": load_fmnist,
    "cifar10": load_cifar10,
}


def convert_data_to_tensor_dataset(data):
    x_tensor = torch.stack([x for x, _ in data])
    y_tensor = torch.tensor([y for _, y in data])
    return TensorDataset(x_tensor, y_tensor)


@click.command()
@click.option("--dataset", type=click.Choice(list(dispatcher.keys())), required=True)
@click.option(
    "--subset_ratio",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1.0,
    help="By default the whole data will be written, Choose a values < 1 to write a subset.",
)
@click.option(
    "--seed",
    type=click.INT,
    default=42,
    help="Random seed is only considered when subset_ratio is less than 1.0.",
)
@click.option("--save_loaded_data_folder", type=click.STRING, default=".tmp")
@click.option("--save_new_data_folder", type=click.STRING, default="clean")
def prepare_data(
    dataset, subset_ratio, seed, save_loaded_data_folder, save_new_data_folder
):
    Path(save_loaded_data_folder).mkdir(parents=True, exist_ok=True)
    Path(save_new_data_folder).mkdir(parents=True, exist_ok=True)

    trainset, testset, train_labels, test_labels = dispatcher[dataset](
        save_loaded_data_folder
    )

    train_ids = np.arange(len(train_labels))
    test_ids = np.arange(len(test_labels))
    if subset_ratio is not None:
        trainset, _, train_ids = subset_selection(
            dataset=trainset, labels=train_labels, ratio=subset_ratio, random_seed=seed
        )
        testset, _, test_ids = subset_selection(
            dataset=testset, labels=test_labels, ratio=subset_ratio, random_seed=seed
        )

    trainset = convert_data_to_tensor_dataset(trainset)
    testset = convert_data_to_tensor_dataset(testset)

    inner_folder_name = (
        f"subset_id{seed}_r{subset_ratio}" if subset_ratio < 1 else f"full"
    )

    folder = Path(save_new_data_folder, dataset, inner_folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    torch.save(trainset, Path(folder, "train.pt"))
    torch.save(testset, Path(folder, "test.pt"))
    save_as_np(train_ids, str(folder), "train_ids.npy")
    save_as_np(test_ids, str(folder), "test_ids.npy")


if __name__ == "__main__":
    prepare_data()
