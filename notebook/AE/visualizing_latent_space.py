import args
import dl_MNIST
import autoencoder

import torch
import numpy as np
from torch.autograd import Variable

# 学習済のモデルを読み込む
autoencoder.model.load_state_dict(torch.load('{}/../../saved_model/AE/autoencoder.pth'.format(dl_MNIST.path), map_location=lambda storage, loc: storage))

test_batchsize = 10000
test_loader = dl_MNIST.DataLoader(dl_MNIST.test_data, batch_size=test_batchsize, shuffle=False)

images, labels = next(iter(test_loader))
images = images.view(test_batchsize, -1).to(autoencoder.device)

# 784次元ベクトルを2次元ベクトルにencode
encoded = autoencoder.model.encoder(Variable(images, volatile=True))
z = encoded.cpu().data.numpy()


import pylab
import matplotlib.pyplot as plt

# Encoderによって生成された潜在空間を可視化
plt.figure(figsize=(10, 10))
plt.scatter(z[:, 0], z[:, 1], marker='.', c=labels.numpy(), cmap=pylab.cm.jet)
plt.colorbar()
plt.grid()
plt.show()