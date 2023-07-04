FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
LABEL authors="y-kwon"

# for avoidance of 'tzdata' configuring
ENV DEBIAN_FRONTEND=noninteractive
# change server to jp ones
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
# for GUI
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Install jupyterlab+.
RUN pip3 install jupyterlab matplotlib==3.3.4 seaborn pandas ipywidgets==7.7.2 tensorboard captum Flask-Compress
# For captum insights
RUN pip3 install gdbgui==0.15.1.0
# for rl-tutorials (dqn only)
RUN apt-get update -qq && apt-get -y install git && rm -rf /var/lib/apt/lists/*
RUN pip3 install gymnasium[classic_control]

ARG uname
ARG password
ARG uid
ARG gid
RUN groupadd -g $gid $uname
RUN useradd -s /bin/bash -u $uid -g $gid -m $uname && echo "$uname:$password" | chpasswd && adduser $uname sudo
RUN apt-get update -qq && apt-get -y install sudo && rm -rf /var/lib/apt/lists/*

WORKDIR /home/$uname
USER $uname

#ENTRYPOINT [""]