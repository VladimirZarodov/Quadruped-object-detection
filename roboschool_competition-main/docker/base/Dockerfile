FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-opencv \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    pkg-config \
    git \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    wget \
    build-essential \
    cmake \
    libpython3.8 \
    libglib2.0-0 \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
