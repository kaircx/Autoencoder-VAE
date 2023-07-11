import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, z_dimention):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, z_dimention)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, z_dimention):
        super().__init__()
        self.layer1 = nn.Linear(z_dimention, 64)
        self.layer2 = nn.Linear(64, 256)
        self.layer3 = nn.Linear(256, output_size)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, z_dimention):
        super().__init__()
        self.layer1 = nn.Linear(z_dimention, 64)
        self.layer2 = nn.Linear(64, 256)
        self.layer3 = nn.Linear(256, output_size)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x

