#!/usr/bin/env python3
"""Retrain dense MNIST model with fc1=140 (10×14, hardware-aligned).
Saves checkpoint to data/checkpoints/mnist_fp32.pt at project root.
"""
import torch, torch.nn as nn, torch.optim as optim, os, random, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
DATA = os.path.join(PROJ, "data")
CKPT = os.path.join(DATA, "checkpoints", "mnist_fp32.pt")

seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 140)   # 140 = 10×14, aligns with systolic array
        self.fc2 = nn.Linear(140, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(root=DATA, train=True, download=False, transform=tf)
test_ds  = datasets.MNIST(root=DATA, train=False, download=False, transform=tf)
train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_ld  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(1, 9):
    model.train()
    for data, target in train_ld:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_ld:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(1).eq(target).sum().item()
            total += len(target)
    acc = 100.0 * correct / total
    print(f"Epoch {epoch}: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc

os.makedirs(os.path.dirname(CKPT), exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "seed": seed,
    "hparams": {"batch_size": 64, "lr": 1e-3, "epochs": 8},
    "best_acc": best_acc,
}, CKPT)
print(f"\nSaved: {CKPT}")
print(f"Best accuracy: {best_acc:.2f}%")
for k, v in model.state_dict().items():
    print(f"  {k}: {v.shape}")
