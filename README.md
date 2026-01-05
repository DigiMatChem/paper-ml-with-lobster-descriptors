[![Testing Linux](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/testing.yml) [![Build Docs](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/docs.yml)

# General
This repository contains code and data used for of the publication
*Assessment of quantum-chemical bonding descriptors predictive ability*


Preprint available at ...

The code has been written by A. Naik with contributions from ....

## Installation

Please follow the steps listed below to be able to successfully create a conda environment to use all the scripts associated with this project. Following the order is important to avoid running into dependencies conflicts.

- Use the provided `dependencies.yml` to create a conda environment using following command `conda env create -f dependencies.yml`. This will create a environment with python v3.10 and required dependencies.
- Activate the newly created conda environment `conda activate ml_env`. Then install SISSO++ with python bindings. Adjust the `CXX` and `CC` paths based on your systems conda env path in command below.
    ```bash
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
    make test
    ```
- Open the `.bashrc` and set `LD_LIBRARY_PATH` environment variable. Do not forget to adjust the path of conda env as per your system.
    ```bash
    export LD_LIBRARY_PATH="/opt/conda/envs/ml_env/lib:$LD_LIBRARY_PATH"
    ```
- Install the rest of python dependencies in the same conda environment.
    ```bash
    pip install arfs==3.0.0 mendeleev==1.1.0 feature-engine==1.9.3 && \
    pip install --upgrade emmet-core --force-reinstall --no-deps && \
    pip install modnet==0.4.5 && \
    pip install lobsterpy==0.5.8 && \
    pip install . && \
    pip cache purge
    ```

Alternatively, one can simply use the [docker image](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/pkgs/container/paper-ml-with-lobster-descriptors%2Fmlproject-python-3.10) associated with this repository to have an working environment setup directly with all required dependencies. For this one can simply open the repository in github codespaces or with vscode. Note for using devcontainer with vscode, one needs docker installed on the system.

## Documentation
Refer the [documentation](https://digimatchem.github.io/paper-ml-with-lobster-descriptors/) for accessing all the results of manuscript. It also consists of scripts to reproduce all the results of the publication and API reference for the codes used.

## Notebooks

### Target data extraction

The scripts in `notebooks/targets` are used to extract the target datasets. The provided notebooks includes comments where need and should self explanatory. Please refer the `notebooks/targets/*/README.md` therein if it exists first before executing the scripts.

### ML scripts

The scripts in `notebooks/ml_scripts` can be used to reproduce the results of the manuscripts. Each subdirectory is named as per sections in the manuscript and includes comments to make it self explanotory and easy to follow.
