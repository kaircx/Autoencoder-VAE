# Autoencoder_VAE
MNISTを用いてAEとVAEを訓練し、双方の潜在空間を比較し考察する

# セットアップ方法

```bash
git clone https://github.com/kaircx/Autoencoder-VAE/tree/main
cd Autoencoder-VAE
cd docker
source build_docker.sh {任意のパスワード}
source run_docker.sh
```
`http://localhost:63333`にアクセス

`notebook`内のJupyter notebookを実行



# 使用しているポートとその用途
※ホストと仮想環境のIPは全部一緒
|ポート|用途|
|---|---|
|63333|jupyter|
|6003|tensorboard用|
|6004|予備|

# docker内への入り方
`docker exec -it m1tutorial_autoencoder_vae bash`

# tokenの確認方法
コンテナ内で
`jupyter server list`
