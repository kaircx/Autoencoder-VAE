import os # tensorboardの出力先作成
import matplotlib.pyplot as plt # 可視化
import numpy as np # 計算
import argparse
import torch # 機械学習フレームワークとしてpytorchを使用
import torch.nn as nn # クラス内で利用するモジュールのため簡略化
import torch.nn.functional as F # クラス内で利用するモジュールのため簡略化
from torch import optim # 最適化アルゴリズム
from torch.utils.tensorboard import SummaryWriter # tensorboardの利用
from torchvision import datasets, transforms # データセットの準備

# GPUが使える場合はGPU上で動かす
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensorboardのログの保存先
path=os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(path+"/../../logs/VAE"):
    os.makedirs(path+"/../../logs/VAE")

    
# argparseを使用してコマンドラインからハイパーパラメータを設定
parser = argparse.ArgumentParser(description='Train a VAE on the MNIST dataset')
parser.add_argument('--z_dim', type=int, default=20, help='Dimension of the latent variables')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for the dataloader')
parser.add_argument('--enc_units', type=int, default=400, help='Number of units in the encoder layers')
parser.add_argument('--dec_units', type=int, default=400, help='Number of units in the decoder layers')
args = parser.parse_args()


# MNISTのデータをとってくるときに一次元化する前処理
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# trainデータとtestデータに分けてデータセットを取得
dataset_train_valid = datasets.MNIST(path+"/../../dataset/", train=True, download=True, transform=transform)
dataset_test = datasets.MNIST(path+"/../../dataset/", train=False, download=True, transform=transform)

# trainデータの20%はvalidationデータとして利用
size_train_valid = len(dataset_train_valid) # 60000
size_train = int(size_train_valid * 0.8) # 48000
size_valid = size_train_valid - size_train # 12000
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train_valid, [size_train, size_valid])

# 取得したデータセットをDataLoader化する
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)


class VAE(nn.Module):
    def __init__(self, z_dim, enc_units, dec_units):
        super(VAE, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = 28 * 28
        self.z_dim = z_dim
        self.enc_fc1 = nn.Linear(self.x_dim, enc_units)
        self.enc_fc2 = nn.Linear(enc_units, enc_units//2)
        self.enc_fc3_mean = nn.Linear(enc_units//2, z_dim)
        self.enc_fc3_logvar = nn.Linear(enc_units//2, z_dim)
        self.dec_fc1 = nn.Linear(z_dim, dec_units)
        self.dec_fc2 = nn.Linear(dec_units, dec_units)
        self.dec_drop = nn.Dropout(p=0.2)
        self.dec_fc3 = nn.Linear(dec_units, self.x_dim)

    def encoder(self, x):
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z):
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        mean, log_var = self.encoder(x.to(device)) # encoder部分
        z = self.sample_z(mean, log_var, device) # Reparameterization trick部分
        y = self.decoder(z.to(device)).to(device) # decoder部分
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var)) # KLダイバージェンス計算
        reconstruction = torch.sum(x.to(device) * torch.log(y.to(device) + self.eps) + (1 - x.to(device)) * torch.log(1 - y.to(device) + self.eps)) # 再構成誤差計算
        return [KL, reconstruction], z, y


# VAEクラスのコンストラクタに潜在変数の次元数を渡す
model = VAE(args.z_dim, args.enc_units, args.dec_units).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
num_epochs = args.epochs
loss_valid = 10 ** 7
loss_valid_min = 10 ** 7
num_no_improved = 0
num_batch_train = 0
num_batch_valid = 0
writer = SummaryWriter(log_dir=path+"/../../logs/VAE")

# 学習開始
for num_iter in range(num_epochs):
    model.train() # 学習前は忘れずにtrainモードにしておく
    for x, t in dataloader_train: # dataloaderから訓練データを抽出する
        lower_bound, _, _ = model(x, device) # VAEにデータを流し込む
        loss = -sum(lower_bound) # lossは負の下限
        model.zero_grad() # 訓練時のpytorchのお作法
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss_train/KL", -lower_bound[0].cpu().detach().numpy(), num_iter + num_batch_train)
        writer.add_scalar("Loss_train/Reconst", -lower_bound[1].cpu().detach().numpy(), num_iter + num_batch_train)
        num_batch_train += 1
    num_batch_train -= 1 # 次回のエポックでつじつまを合わせるための調整

    # 検証開始
    model.eval() # 検証前は忘れずにevalモードにしておく
    loss = []
    for x, t in dataloader_valid: # dataloaderから検証データを抽出する
        lower_bound, _, _ = model(x, device) # VAEにデータを流し込む
        loss.append(-sum(lower_bound).cpu().detach().numpy())
        writer.add_scalar("Loss_valid/KL", -lower_bound[0].cpu().detach().numpy(), num_iter + num_batch_valid)
        writer.add_scalar("Loss_valid/Reconst", -lower_bound[1].cpu().detach().numpy(), num_iter + num_batch_valid)
        num_batch_valid += 1
    num_batch_valid -= 1 # 次回のエポックでつじつまを合わせるための調整
    loss_valid = np.mean(loss)
    loss_valid_min = np.minimum(loss_valid_min, loss_valid)
    print(f"[EPOCH{num_iter + 1}] loss_valid: {int(loss_valid)} | Loss_valid_min: {int(loss_valid_min)}")

    # もし今までのlossの最小値よりも今回のイテレーxションのlossが大きければカウンタ変数をインクリメントする
    if loss_valid_min < loss_valid:
        num_no_improved += 1
        print(f"{num_no_improved}回連続でValidationが悪化しました")
    # もし今までのlossの最小値よりも今回のイテレーションのlossが同じか小さければカウンタ変数をリセットする
    else:
        num_no_improved = 0
        if not os.path.exists(path+"/../../saved_model/VAE"):
            os.makedirs(path+"/../../saved_model/VAE")
        torch.save(model.state_dict(), path+f"/../../saved_model/VAE/z_{model.z_dim}.pth")
    # カウンタ変数が10回に到達したらearly stopping
    if (num_no_improved >= 10):
        print(f"{num_no_improved}回連続でValidationが悪化したため学習を止めます")
        break

# tensorboardのモニタリングも停止しておく
writer.close()