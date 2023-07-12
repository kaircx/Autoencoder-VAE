import dl_MNIST
import args

import torch
import torch.nn as nn
from torch import optim

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

class AutoEncoder(nn.Module):
    def __init__(self, default_size, z_dimention):
        super().__init__()
        self.encoder = Encoder(default_size, z_dimention)
        self.decoder = Decoder(default_size, z_dimention)
        
    def forward(self, x):
        result_encoder = self.encoder(x)
        result_decoder = self.decoder(result_encoder)
        return result_decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cudaが利用可能ならGPU,そうでないならcpu
model = AutoEncoder(28*28, args.dimention).to(device)
criterion = nn.BCELoss() # 誤差関数
number_of_epochs = args.epochs # エポック数
optimizer = optim.Adam(model.parameters(), lr = 0.001) # OptimizerとしてAdam(学習率: 0.001)を採用

loss_values = []
for an_epoch in range(number_of_epochs):
    loss_sum = 0.0
    count = 0
    for input, t in dl_MNIST.train_loader:
        optimizer.zero_grad()
        
        model.train()
        input = input.view(input.size(0), -1)
        input = input.to(device)
        result_autoencoder = model(input)
        loss = criterion(result_autoencoder, input)
        loss_sum = loss_sum + loss
        
        loss.backward()
        optimizer.step()
        count = count + 1
    loss_average = loss_sum / (count+1)
    loss_values.append(loss_average)
    print('epoch [{}/{}], loss: {:.4f}'.format(an_epoch + 1, number_of_epochs, loss_values[an_epoch]))

torch.save(model.state_dict(), './autoencoder.pth') # 学習済のモデルを保存する