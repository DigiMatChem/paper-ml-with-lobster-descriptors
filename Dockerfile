# Use an official micromamba image as the base image
ARG PYTHON_VERSION=3.10

FROM mambaorg/micromamba:1.5.10


# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_NO_LOW_SPEED_LIMIT=1

# Switch to root to install all dependencies (using non-root user causes permission issues)
USER root

# Make arg accessible to the rest of the Dockerfile
ARG PYTHON_VERSION

# Install system dependencies
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

# Create a micromamba environment using the dependencies listed in dependencies.yml in the current directory
COPY environment.yml /tmp/environment.yml
RUN micromamba create -n ml_env -f /tmp/environment.yml -y && \
    micromamba clean -a -y
ENV PATH=/opt/conda/envs/ml_env/bin:$PATH
SHELL ["micromamba", "run", "-n", "ml_env", "/bin/bash", "-c"]

# Install SISSO++ from git repository with specific version
RUN git clone --recursive https://gitlab.com/sissopp_developers/sissopp.git /opt/sissopp && \
    cd /opt/sissopp && \
    git checkout v1.2.6 && \
    mkdir build && \
    cp cmake/toolchains/gnu_param_py.cmake build/ && \
    echo 'set(SISSO_BUILD_TESTS ON CACHE BOOL "")' >> build/gnu_param_py.cmake && \
    ./build_third_party.bash -j 4 && \
    cd build && \
    cmake -C gnu_param_py.cmake ../ && \
    make -j 4 && \
    make install
    make test

# Set the working directory
WORKDIR /workspace
# Activate the micromamba environment by default when starting a container
CMD ["micromamba", "run", "-n", "ml_env", "/bin/bash"]
