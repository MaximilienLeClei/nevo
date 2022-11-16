FROM ubuntu:20.04
COPY sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    libopenmpi-dev

COPY requirements.txt /nevo/requirements.txt

RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/Downloads/
RUN mkdir -p ~/.mujoco/ && tar -zxf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/