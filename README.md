# π-fuzz: Metamorphic Action Policy Testing Framework

π-fuzz is a metamorphic action policy testing framework based on fuzz testing. Refer to the original AAAI paper for more detail about the approach. Below, we provide instructions for installing dependencies and replicating the results on a newly created Docker image. Refer to this link do download a ready-to-use Docker image.

## Setup
Here are the 10 steps of making your working environment ready to π-fuzz:

1. After creating a Ubuntu 20.04 installed Docker image, verify that `Python3.8` is installed, if not install.
2. Install `pip` package manager by following instructions [here](https://pip.pypa.io/en/stable/installation/).
3. Install `Cython` as follows: `pip install cython`
4. Install other required packages as follows: `apt-get install gcc python3-dev python3-dev g++`
5. Install `conda` package manager by following instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
6. Clone the repository in a desired directory and enter (Skip this step if you are using the supplementary file provided together with the paper). 
7. Run `git submodule init` and `git submodule update` commands to receive dependant submodules (Skip this step if you are using the supplementary file provided together with the paper).
8. Create a new conda environment with the follwoing command `conda env create -f environment.yml`.
9. Upon creation activate the environment with `conda activate pi-fuzz`.
10. Create a directory named `logs` and you are good to go.

## Getting Started 

`fuzz_config.py` file contains the two main parameters, `FUZZ_BUDGET` and `SEARCH_BUDGET`, used throught all experiments presented in the paper. You can change them according to your wish. Try running the following command after setting those parameters to smaller values (e.g. `FUZZ_BUDGET=10` and `SEARCH_BUDGET=10`).

`python run.py -E lunar -R 123 -A policies/lunar_org -F inc -CT 0.6 -FMB 25`

Upon completion a pickle file should be created that contains all information regarding the fuzzing campaign (e.g. pool size, number of bugs etc.). Together with this a log file is created under `logs` folder.  

A number of parameters needed to be set to start a fuzzing campaing. Below we provide details:

| Parameter  | Options       | Required | Explanation |
| :---------: |-------------| :-----:| -------------  |
| E  | linetrack, lunar, bipedal  | Yes | Environment |
| A  | - | Yes | Path to agent  |
| R  | - | Yes | Random seed  |
| F  | inc, non-inc  | No | Fuzzing type |
| CT | - | No  | Coverage threshold |
| FMB | - | No | Fuzz mutation budget |
| IP | - | No | Informed mutations probability |


## Replicating Results

We test π-fuzz on 3 domains: `linetrack`, `lunar` and `bipedal`. For each domain, we consider 6 fuzzer settings in which we change fuzzing type and informed mutations probabilities. For example, for `lunar` domain we run the following experiments:

1. `python run.py -E lunar -R 1 -A policies/lunar_org -F inc -CT 0.6 -FMB 25 -IP 0`
2. `python run.py -E lunar -R 1 -A policies/lunar_org -F inc -CT 0.6 -FMB 25 -IP 0.1`
3. `python run.py -E lunar -R 1 -A policies/lunar_org -F inc -CT 0.6 -FMB 25 -IP 0.2`
4. `python run.py -E lunar -R 1 -A policies/lunar_org -F non-inc -CT 0.6 -FMB 25 -IP 0`
5. `python run.py -E lunar -R 1 -A policies/lunar_org -F non-inc -CT 0.6 -FMB 25 -IP 0.1`
6. `python run.py -E lunar -R 1 -A policies/lunar_org -F non-inc -CT 0.6 -FMB 25 -IP 0.2`

For running experiments on other domains, provide the corresponding domain keyword after `-E`. For replicating the exact results, run each experiment with random seeds from 1 to 8. Remember to keep `fuzz_config.py` file as default (i.e. `FUZZ_BUDGET=86400` and `SEARCH_BUDGET=1000`). `fuzz_utils.py` provides some useful functions to plot results (i.e. poolsize_over_time, warn_over_time, boxplot).


