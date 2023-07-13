import args

import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# MNISTの取得
transform = T.Compose([T.ToTensor()])

#バッチサイズの指定
batchsize = args.batch_size

# パスの指定
path = os.path.dirname(os.path.abspath(__file__))

# 学習データ
train_dataset = torchvision.datasets.MNIST(root='{}/../../dataset/'.format(path), train=True,download=True,transform=transform)

# 学習データの80%を訓練用に,20%を検証用に分ける
length_of_dataset = len(train_dataset)
length_of_train_data = int(length_of_dataset * 0.8)
length_of_valid_data = length_of_dataset - length_of_train_data
train_data, valid_data = torch.utils.data.random_split(train_dataset, [length_of_train_data, length_of_valid_data])

# 訓練データと検証データのロード
train_loader = DataLoader(train_data, batch_size = batchsize, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size = batchsize)

# テストデータ
test_data = torchvision.datasets.MNIST(root='{}/../../dataset/'.format(path), train=False,download=True,transform=transform)
test_loader = DataLoader(test_data, batch_size = batchsize)
