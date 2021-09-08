### π-fuzz: Metamorphic Action Policy Testing Framework

π-fuzz is a metamorphic action policy testing framework based on fuzz testing. Refer to the original AAAI paper for more detail about the approach. Below, we provide instructions for installing dependencies and replicating the results on a newly created Docker image. Refer to this link do download a ready-to-use Docker image.

## Setup
Here are the 10 steps of making your working environment ready to π-fuzz:

1. After creating a Ubuntu 20.04 installed Docker image, verify that `Python3.8` is installed, if not install.
2. Install `pip` package manager by following instructions [here](https://pip.pypa.io/en/stable/installation/).
3. Install `Cython` as follows: `pip install cython`
4. Install other required packages as follows: `apt-get install gcc python3-dev python3-dev g++`
5. Install `conda` package manager by following instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
6. Clone the repository in a desired directory and enter. 
7. Run `git submodule init` and `git submodule update` commands to receive dependant submodules.
8. Create a new conda environment with the follwoing command `conda env create -f environment.yml`.
9. Upon creation activate the environment with `conda activate pi-fuzz`.
10. Create a directory named `logs` and you are good to go.
