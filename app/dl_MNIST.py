import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# MNISTの取得
transform = T.Compose([T.ToTensor()])

path = os.path.dirname(os.path.abspath(__file__))

# 学習データ
train_data = torchvision.datasets.MNIST(root='{}/../dataset/'.format(path), train=True,download=True,transform=transform)
train_loader = DataLoader(train_data,batch_size = 64)

# テストデータ
test_data = torchvision.datasets.MNIST(root='{}/../dataset/'.format(path), train=False,download=True,transform=transform)
test_loader = DataLoader(test_data,batch_size = 64)
