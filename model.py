import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BaseEncoder(nn.Module):

    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(0, 120)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Calculate the input size for fc1
        # fc1_input_size = x.shape[1] * x.shape[2] * x.shape[3]
        # Initialize fc1 if it hasn't been initialized yet or if the input size has changed
        # if self.fc1 is None or fc1_input_size != self.fc1.in_features:
        #   self.fc1 = nn.Linear(fc1_input_size, 120)
        # x = x.view(-1, fc1_input_size)
        x = x.view(-1, 16 * 4 * 4)
        # x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(84, 512)  # First fully connected layer
        # self.fc1 = nn.Linear(1000, 512)  # For resnet18
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = F.relu(self.fc2(x))
        return x

class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(256, 10)  # Fully connected layer

    def forward(self, x):
        x = F.log_softmax(self.fc(x), dim=1)  # Apply fully connected layer
        return x


class MOON(nn.Module):
    def __init__(self):
        super(MOON, self).__init__()
        self.base_encoder = BaseEncoder()
        # self.base_encoder = models.resnet18(pretrained=False)
        # self.base_encoder = models.resnet50(pretrained=False)
        self.projection_head = ProjectionHead()
        self.output_layer = OutputLayer()
        # self.output_layer = nn.Linear(256, 10)


    def forward(self, x):
        x = self.base_encoder(x)
        x = self.projection_head(x)
        x = self.output_layer(x)
        return x

    def Rw(self, x):
        x = self.base_encoder(x)
        x = self.projection_head(x)
        return x