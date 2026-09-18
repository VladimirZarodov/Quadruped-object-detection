FROM aliengo-isaac-gym

WORKDIR /workspace/aliengo_competition

COPY . /workspace/aliengo_competition

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install numpy matplotlib tensorboard \
    && python -m pip install git+https://github.com/leggedrobotics/rsl_rl.git@v1.0.2 \
    && python -m pip install -e .

ENV PYTHONPATH=/opt/isaacgym/python:/workspace/aliengo_competition/src:/workspace/aliengo_competition
