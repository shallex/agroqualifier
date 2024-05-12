import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, size, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten_size = (32 * size[0] * size[1]) // (2 * 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.flatten_size, 512) 
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x