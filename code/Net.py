import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2_bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2_bn2 = nn.BatchNorm2d(32)

        # First fully connected layer
        self.fc1 = nn.Linear(9248, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        # Second fully connected layer that outputs labels
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv2(x)), 2))

        # Flatten x
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1_bn(self.fc1(x)))

        output = self.fc2(x)
        return output

