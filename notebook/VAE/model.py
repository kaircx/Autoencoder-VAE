import os # tensorboardの出力先作成
import matplotlib.pyplot as plt # 可視化
import numpy as np # 計算
import torch # 機械学習フレームワークとしてpytorchを使用
import torch.nn as nn # クラス内で利用するモジュールのため簡略化
import torch.nn.functional as F # クラス内で利用するモジュールのため簡略化
from torch import optim # 最適化アルゴリズム
from torch.utils.tensorboard import SummaryWriter # tensorboardの利用
from torchvision import datasets, transforms # データセットの準備

# GPUが使える場合はGPU上で動かす
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensorboardのログの保存先
if not os.path.exists("../logs"):
    os.makedirs("../logs")

# MNISTのデータをとってくるときに一次元化する前処理
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# trainデータとtestデータに分けてデータセットを取得
dataset_train_valid = datasets.MNIST("./", train=True, download=True, transform=transform)
dataset_test = datasets.MNIST("./", train=False, download=True, transform=transform)

# trainデータの20%はvalidationデータとして利用
size_train_valid = len(dataset_train_valid) # 60000
size_train = int(size_train_valid * 0.8) # 48000
size_valid = size_train_valid - size_train # 12000
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train_valid, [size_train, size_valid])

# 取得したデータセットをDataLoader化する
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1000, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=False)

class VAE(nn.Module):
    def __init__(self, z_dim):
        """コンストラクタ

        Args:
            z_dim (int): 潜在空間の次元数

        Returns:
            None.

        Note:
            eps (float): オーバーフローとアンダーフローを防ぐための微小量
        """
        super(VAE, self).__init__() # VAEクラスはnn.Moduleを継承しているため親クラスのコンストラクタを呼ぶ必要がある
        self.eps = np.spacing(1) # オーバーフローとアンダーフローを防ぐための微小量
        self.x_dim = 28 * 28 # MNISTの場合は28×28の画像であるため
        self.z_dim = z_dim # インスタンス化の際に潜在空間の次元数は自由に設定できる
        self.enc_fc1 = nn.Linear(self.x_dim, 400) # エンコーダ1層目
        self.enc_fc2 = nn.Linear(400, 200) # エンコーダ2層目
        self.enc_fc3_mean = nn.Linear(200, z_dim) # 近似事後分布の平均
        self.enc_fc3_logvar = nn.Linear(200, z_dim) # 近似事後分布の分散の対数
        self.dec_fc1 = nn.Linear(z_dim, 200) # デコーダ1層目
        self.dec_fc2 = nn.Linear(200, 400) # デコーダ2層目
        self.dec_drop = nn.Dropout(p=0.2) # 過学習を防ぐために最終層の直前にドロップアウト
        self.dec_fc3 = nn.Linear(400, self.x_dim) # デコーダ3層目

    def encoder(self, x):
        """エンコーダ

        Args:
            x (torch.tensor): (バッチサイズ, 入力次元数)サイズの入力データ

        Returns:
            mean (torch.tensor): 近似事後分布の平均
            logvar (torch.tensor): 近似事後分布の分散の対数
        """
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean, log_var, device):
        """Reparameterization trickに基づく潜在変数Zの疑似的なサンプリング

        Args:
            mean (torch.tensor): 近似事後分布の平均
            logvar (torch.tensor): 近似事後分布の分散の対数
            device (String): GPUが使える場合は"cuda"でそれ以外は"cpu"

        Returns:
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数
        """
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z):
        """デコーダ

        Args:
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数

        Returns:
            y (torch.tensor): (バッチサイズ, 入力次元数)サイズの再構成データ
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        """順伝播処理

        Args:
            x (torch.tensor): (バッチサイズ, 入力次元数)サイズの入力データ
            device (String): GPUが使える場合は"cuda"でそれ以外は"cpu"

        Returns:
            KL (torch.float): KLダイバージェンス
            reconstruction (torch.float): 再構成誤差
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数
            y (torch.tensor): (バッチサイズ, 入力次元数)サイズの再構成データ            
        """
        mean, log_var = self.encoder(x.to(device)) # encoder部分
        z = self.sample_z(mean, log_var, device) # Reparameterization trick部分
        y = self.decoder(z.to(device)).to(device) # decoder部分
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var)) # KLダイバージェンス計算
        reconstruction = torch.sum(x.to(device) * torch.log(y.to(device) + self.eps) + (1 - x.to(device)) * torch.log(1 - y.to(device) + self.eps)) # 再構成誤差計算
        return [KL, reconstruction], z, y


