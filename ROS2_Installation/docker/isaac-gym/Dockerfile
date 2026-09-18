FROM aliengo-base

WORKDIR /opt

COPY isaacgym /opt/isaacgym

# torch
RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# isaacgym
WORKDIR /opt/isaacgym/python
RUN pip install -e .
RUN python -c "import isaacgym"
ENV PYTHONPATH=/opt/isaacgym/python
WORKDIR /workspace
