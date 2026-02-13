[![Testing Linux](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/testing.yml) [![Build Docs](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/actions/workflows/docs.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18599928.svg)](https://doi.org/10.5281/zenodo.18599928)

# General

`mlproject` is a package that hosts all the utility scripts to reproduce the results from our publication: **A critical assessment of bonding descriptors for predicting materials properties**.

To keep the publication concise, not all results are included. This repository provides a one-stop access point to all results, along with all code and data required to reproduce them. Results are organized as static HTML pages for easy navigation and are deployed on github pages.


Preprint available at [https://doi.org/10.48550/arXiv.2602.12109](https://doi.org/10.48550/arXiv.2602.12109).

The code has been written by A. Naik with contributions from Dr. Philipp Benner.


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

## Results and API reference
Refer the [rendered website](https://digimatchem.github.io/paper-ml-with-lobster-descriptors/) for accessing all the results of manuscript. It also consists of scripts to reproduce all the results of the publication and API reference for the codes used.

## Notebooks

### Target data extraction

The scripts in [notebooks/targets](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/tree/main/notebooks/targets) are used to extract the target datasets. The provided notebooks includes comments where need and should self explanatory. Please refer the `notebooks/targets/*/README.md` therein if it exists first before executing the scripts.

### ML scripts

The scripts in [notebooks/ml_scripts](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/tree/main/notebooks/ml_scripts) can be used to reproduce the results of the manuscript. Each subdirectory is named as per sections in the manuscript and includes comments to make it self explanotory and easy to follow.

### Misc

The script in [notebooks/misc](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors/tree/main/notebooks/misc) can be used to generate the correlation plots shown in the manuscript.


## ML models

To support reproducibility, the trained machine learning models and post-processed data used to generate the figures in the publication are made available via Zenodo at the following DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18329982.svg)](https://doi.org/10.5281/zenodo.18329982)


## Other important citations

### Code
- Projected force constant
   - [JanHempelmann/projectionscript](https://github.com/JanHempelmann/projectionscript.git) (reference code repository, currently private)
- Lattice thermal conductivity
   - [clean_data](https://github.com/masato1122/phonon_e3nn/blob/799c0f65e9b8cc01afbe7647fceb8ba07da3058b/tools/run_prediction.py#L37C1-L70C1) function

### Data

- Vibration and thermodynamic properties
  - [High-throughput density-functional perturbation theory phonons for inorganic materials](http://www.nature.com/articles/sdata201865)
  - Data available for download [here](https://springernature.figshare.com/collections/High-throughput_Density-Functional_Perturbation_Theory_phonons_for_inorganic_materials/3938023)
- Lattice thermal conductivity
  - [Database and deep-learning scalability of anharmonic phonon properties by automated brute-force first-principles calculations](https://doi.org/10.48550/arXiv.2504.21245)
  - Data available for download [here](https://github.com/masato1122/phonon_e3nn/tree/799c0f65e9b8cc01afbe7647fceb8ba07da3058b)
- Elasticity
  - [Commentary: The Materials Project: A materials genome approach to accelerating materials innovation](https://doi.org/10.1063/1.4812323)

### Other related works
- Projected force constant
   - [Vibrational properties and bonding nature of Sb2Se3 and their implications for chalcogenide materials](https://doi.org/10.1039/C5SC00825E)
   - [Long-Range Forces in Rock-Salt-Type Tellurides and How they Mirror the Underlying Chemical Bonding](https://doi.org/10.1002/adma.202100163)
