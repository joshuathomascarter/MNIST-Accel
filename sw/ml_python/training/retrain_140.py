#!/usr/bin/env python3
"""Quick retrain of MNIST CNN with fc1=140 (10×14) to match hardware tiling.
Stops as soon as test accuracy hits 99%. Saves to data/checkpoints/mnist_fp32.pt"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, sys

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 140)  # 140 = 10×14, tiles perfectly
        self.fc2 = nn.Linear(140, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA = os.path.join(ROOT, 'data')
    CKPT = os.path.join(DATA, 'checkpoints', 'mnist_fp32.pt')
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root=DATA, train=True, download=True, transform=tf)
    test_ds  = datasets.MNIST(root=DATA, train=False, download=False, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    torch.manual_seed(42)
    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, 16):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(data), target)
            loss.backward()
            opt.step()
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} [{batch_idx*64}/{len(train_dl.dataset)}] loss={loss.item():.4f}")

        scheduler.step()

        # Test
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_dl:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()
        acc = 100.0 * correct / len(test_ds)
        print(f"Epoch {epoch}: accuracy = {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'state_dict': model.state_dict(),
                'seed': 42,
                'hparams': {'fc1_width': 140, 'block_size': 14},
                'best_acc': best_acc,
            }, CKPT)
            print(f"  -> Saved checkpoint ({best_acc:.2f}%)")

        if acc >= 99.0:
            print(f"\nReached {acc:.2f}% >= 99% target. Done!")
            break

    print(f"\nFinal best: {best_acc:.2f}%")
    print(f"Saved to: {CKPT}")

    # Verify shapes
    c = torch.load(CKPT, map_location='cpu', weights_only=False)
    for k, v in c['state_dict'].items():
        print(f"  {k}: {list(v.shape)}")

if __name__ == '__main__':
    main()
