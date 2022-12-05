"""Trying to make a MNSIT web app"""
from __future__ import annotations

from torch.nn import CrossEntropyLoss, Linear, Flatten, Conv2d , MaxPool2d, Module, ReLU
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torch

class MNISTDataset:
    """MNIST Dataset with loaders for training and testing"""
    def __init__(self, batch_size: int = 128, num_workers: int = 0,pin_memory: bool = False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = MNIST(root="./.dataset", train=True, download=True, transform=T.ToTensor())
        self.test_dataset = MNIST(root="./.dataset", train=False, download=True, transform=T.ToTensor())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,pin_memory = self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory = self.pin_memory
        )

class MNISTModel(Module):
    """Mnist model with 2 convolutional layers and 3 linear layers"""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(1,128,5)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(2)
        self.conv2 = Conv2d(128,256,5)
        self.flatten = Flatten()
        self.linear1 = Linear(4096, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 10)
    

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the model"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


def step(model : Module , optimizer : AdamW, criterion, dataloader : DataLoader, nb_epoch : int = 10,train : bool = True) -> None:
    """Training and testing step"""
    total_acc,total_loss = 0.0,0.0
    for i in range(1,nb_epoch+1):
        print(f"epoch nb : {i}")
        for data,labels in dataloader:
            _output = model(data)
            _acc = (labels == _output.argmax(dim=1)).sum()
            _loss = criterion(_output,labels)

        if(train):
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
        total_acc += _acc.item()/len(dataloader.dataset)
        total_loss += _loss.item()/len(dataloader)
        display_acc = (total_acc*100)/i
        print(f"total acc : {display_acc:.2f}",f"total loss : {total_loss:.2f}")
        


if __name__ == "__main__":
    datasets = MNISTDataset(num_workers=6)
    train_loader = datasets.train_dataloader()
    test_loader = datasets.test_dataloader()
    criterion = CrossEntropyLoss()
    model = MNISTModel()
    otpimizer = AdamW(model.parameters(), lr=1e-3)
    print("Training")
    step(model, otpimizer, criterion, train_loader, nb_epoch=10)
    print("Testing")
    step(model, otpimizer, criterion, test_loader, nb_epoch=1,train=False)