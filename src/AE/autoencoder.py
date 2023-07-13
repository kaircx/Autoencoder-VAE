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

class AutoEncoder(nn.Module):
    def __init__(self, default_size, z_dimention):
        super().__init__()
        self.encoder = Encoder(default_size, z_dimention)
        self.decoder = Decoder(default_size, z_dimention)
        
    def forward(self, x):
        result_encoder = self.encoder(x)
        result_decoder = self.decoder(result_encoder)
        return result_decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cudaが利用可能ならGPU,そうでないならcpuをセット
model = AutoEncoder(28*28, args.dimention).to(device)
criterion = nn.BCELoss() # 誤差関数として交差エントロピー誤差を採用
number_of_epochs = args.epochs # エポック数
optimizer = optim.Adam(model.parameters(), lr = 0.001) # OptimizerとしてAdam(学習率: 0.001)を採用

loss_train_values = []
loss_valid_values = []

loss_by_batch = []
for an_epoch in range(number_of_epochs):
    count = 0
    loss_train_sum = 0.0
    model.train() # 訓練モードに変更する
    for image, label in train_loader:
        optimizer.zero_grad()
        
        image = image.view(image.size(0), -1)
        image = image.to(device)
        result_autoencoder = model(image)
        loss = criterion(result_autoencoder, image) #入力画像と出力画像の誤差を計算する
        loss_train_sum = loss_train_sum + loss
        
        loss.backward()
        optimizer.step()

        # バッチごとの誤差
        loss_by_batch.append(loss.data)
        
        count = count + 1

    loss_train_average = loss_train_sum / count   
    loss_train_values.append(loss_train_average)
    print('epoch [{}/{}], train_loss: {:.4f}'.format(an_epoch + 1, number_of_epochs, loss_train_values[an_epoch]))

    count = 0
    loss_valid_sum = 0.0
    model.eval() # 検証モードに変更
    for image, label in valid_loader:
        image = image.view(image.size(0), -1)
        image = image.to(device)
        result_autoencoder = model(image)
        loss = criterion(result_autoencoder, image) #入力画像と出力画像の誤差を計算する
        loss_valid_sum = loss_valid_sum + loss

        count = count + 1
        
    loss_valid_average = loss_valid_sum / count   
    loss_valid_values.append(loss_valid_average)
    print('epoch [{}/{}], valid_loss: {:.4f}'.format(an_epoch + 1, number_of_epochs, loss_valid_values[an_epoch]))

## 学習済のモデルを保存する ##
if not os.path.exists("{}/../saved_model/AE/".format(dl_MNIST,path)):
    os.makedirs("{}/../saved_model/AE/".format(dl_MNIST,path))
torch.save(model.state_dict(), '{}/../saved_model/AE/autoencoder.pth'.format(dl_MNIST.path))
