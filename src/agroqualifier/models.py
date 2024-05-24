import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101


class SimpleCNN(nn.Module):
    def __init__(self, params):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(params.model_params.input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten_size = (32 * params.dataset_params.size[0] * params.dataset_params.size[1]) // (2 * 2)
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


class ResNet101_3_L(nn.Module):
    def __init__(self, params):
        super(ResNet101_3_L, self).__init__()
        self.resnet = resnet101(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, params.model_params.output_channels),
            torch.nn.ReLU()
        )
    def forward(self, x):
        return self.resnet(x)

class ResNet101_2_L(nn.Module):
    def __init__(self, params):
        super(ResNet101_2_L, self).__init__()
        self.resnet = resnet101(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, params.model_params.output_channels),
        torch.nn.ReLU()
        )
    def forward(self, x):
        return self.resnet(x)