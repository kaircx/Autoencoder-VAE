#!/bin/bash

PARENT_DIR=$(pwd|xargs dirname)

docker run --name m1tutorial_autoencoder_vae -it --rm -d --gpus all --ipc=host \
           --ulimit memlock=-1 --ulimit stack=67108864 \
           -v /tmp/.X11-unix:/tmp/.X11-unix: -v "${PARENT_DIR}":/home/"${USER}"/Autoencoder-VAE \
           -e DISPLAY="${DISPLAY}" \
           -p 63333:63333 \
           -p 6003:6003 \
           -p 6004:6004 \
           torch_on_jupyter jupyter lab --config='./Autoencoder-VAE/jupyter_lab_config.py'
