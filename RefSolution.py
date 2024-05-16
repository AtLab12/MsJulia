import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from memory_profiler import memory_usage
import psutil

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1014, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print("Accuracy of the network on the test images: %d %%" % accuracy)


def measure_time_and_memory(func):
    start_time = time.time()
    mem_usage = memory_usage((func,))
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    print(f"Memory usage: {max(mem_usage) - min(mem_usage)} MiB")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(3):
        print(f"Starting epoch {epoch+1}")

        def train_func():
            train_model(model, train_loader, criterion, optimizer, device)

        def eval_func():
            evaluate_model(model, test_loader, device)

        print(
            "Before training epoch - Current CPU memory usage:",
            psutil.Process().memory_info().rss / (1024**2),
            "MiB",
        )
        measure_time_and_memory(train_func)
        print(
            "After training epoch - Current CPU memory usage:",
            psutil.Process().memory_info().rss / (1024**2),
            "MiB",
        )
        measure_time_and_memory(eval_func)
