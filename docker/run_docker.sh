#!/bin/bash

PARENT_DIR=$(pwd|xargs dirname)

docker run --name torch_on_jupyter -it --rm -d --gpus all --ipc=host \
           --ulimit memlock=-1 --ulimit stack=67108864 \
           -v /tmp/.X11-unix:/tmp/.X11-unix: -v "${PARENT_DIR}":/home/"${USER}"/TorchJupyterEnv \
           -e DISPLAY="${DISPLAY}" \
           -p 63322:63322 \
           -p 6006:6006 \
           -p 6007:6007 \
           torch_on_jupyter jupyter lab --config='./TorchJupyterEnv/jupyter_lab_config.py'
