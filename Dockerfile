# Use an official micromamba image as the base image

ARG PYTHON_VERSION=3.10
FROM mambaorg/micromamba:1.5.10

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_NO_LOW_SPEED_LIMIT=1

ARG PYTHON_VERSION

# --- Switch to root to install system dependencies ---
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    bc \
    ffmpeg \
    unzip \
    wget \
    gfortran \
    liblapack-dev \
    libblas-dev \
    cmake \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Create a non-root user ---
ARG USERNAME=mluser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g ${USER_GID} ${USERNAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}

# Switch to the non-root user
USER ${USERNAME}
WORKDIR /workspace

# --- Create micromamba environment ---
COPY --chown=${USERNAME}:${USERNAME} dependencies.yml /tmp/environment.yml
RUN micromamba create -n ml_env -f /tmp/environment.yml -y && \
    micromamba clean -a -y

ENV PATH=/opt/conda/envs/ml_env/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/ml_env/lib:$LD_LIBRARY_PATH

SHELL ["micromamba", "run", "-n", "ml_env", "/bin/bash", "-c"]

# --- Install SISSO++ ml_env conda env and store it in the user's home directory ---
RUN micromamba run -n ml_env bash -c '\
git clone --recursive https://gitlab.com/sissopp_developers/sissopp.git ~/sissopp && \
cd ~/sissopp && \
git checkout v1.2.6 && \
mkdir build && \
cp cmake/toolchains/gnu_param_py.cmake build/ && \
echo "set(SISSO_BUILD_TESTS ON CACHE BOOL \"\")" >> build/gnu_param_py.cmake && \
./build_third_party.bash CXX=/opt/conda/envs/ml_env/bin/g++ CC=/opt/conda/envs/ml_env/bin/gcc -j4 && \
cd build && \
cmake -C gnu_param_py.cmake ../ && \
make -j4 && \
make install && \
make test'

# Copy project files into workspace
COPY --chown=${USERNAME}:${USERNAME} . /workspace

# Ensure the default shell uses micromamba and activates the ml_env environment
SHELL ["micromamba", "run", "-n", "ml_env", "/bin/bash", "-c"]

# Default command to start the container with the environment active
CMD ["micromamba", "run", "-n", "ml_env", "/bin/bash"]

# Install pending dependencies
RUN micromamba run -n ml_env bash -c '\
pip install arfs==3.0.0 mendeleev==1.1.0 feature-engine==1.9.3 && \
pip install --upgrade emmet-core --force-reinstall --no-deps && \
pip install modnet==0.4.5 && \
pip install git+https://github.com/JaGeo/LobsterPy.git && \
pip install . && \
pip cache purge'
