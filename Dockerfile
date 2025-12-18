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

SHELL ["micromamba", "run", "-n", "ml_env", "/bin/bash", "-c"]

# --- Install SISSO++ in the user's home directory ---
RUN git clone --recursive https://gitlab.com/sissopp_developers/sissopp.git ~/sissopp && \
    cd ~/sissopp && \
    git checkout v1.2.6 && \
    mkdir build && \
    cp cmake/toolchains/gnu_param_py.cmake build/ && \
    echo 'set(SISSO_BUILD_TESTS ON CACHE BOOL "")' >> build/gnu_param_py.cmake && \
    sed -i 's/\*"g++")/*"g++"*|*"x86_64-conda-linux-gnu-c++"*)/g' build_third_party.bash && \
    ./build_third_party.bash -j 4 && \
    cd build && \
    cmake -C gnu_param_py.cmake ../ && \
    make -j 4 && \
    make install && \
    make test

# Copy project files into workspace
COPY --chown=${USERNAME}:${USERNAME} . /workspace

# Default command to run container with micromamba environment
CMD ["micromamba", "run", "-n", "ml_env", "/bin/bash"]
