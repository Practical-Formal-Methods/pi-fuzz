# π-fuzz: Metamorphic Action Policy Testing Framework

π-fuzz is a metamorphic action policy testing framework implemented as part of our ISSTA 2022 paper _Metamorphic Relations via Relaxations: An Approach to Obtain Oracles for Action-Policy Testing_. In this document, we provide instructions for installing dependencies for running π-fuzz and replicating the results.

## Setup
Here are the 8 steps of making your working environment ready for π-fuzz:

1. After creating a Ubuntu 20.04 installed Docker image, 
2. Ensure that `Python3` and `pip3` are installed.
3. Install `conda` package manager by following instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
4. Clone the repository in a desired directory and enter in it. 
5. Create a new conda environment with the following command `conda env create -f environment.yml`.
6. Upon creation activate the environment with `conda activate pi-fuzz`.

Now you are ready to run π-fuzz.

## Getting Started 
`run.py` is the entry point to π-fuzz. It first launches the fuzzer and then calls the oracle to identify bugs with given configurations. Below parameters determines π-fuzz configurations along with `config.py` file. The settings in `config.py` file were kept the same accross all experiments presentd in the paper. 

| Parameter  | Options     | Default | Required | Explanation |
| :---------: |-------------| ----- | :-------------:  | -----|
| E  | highway, lunar, bipedal | - | Yes | Environment |
| A  | - | - |Yes | Path to agent-to-test  |
| R  | **int** | - |Yes | Random seed  |
| L  | - | pifuzz_logs | No | Path to log folder |
| F  | inc, non-inc  | inc |No | Fuzzing type |
| IP | 0 < **float** < 1| 0.2 | No | Informed mutations probability |
| O  | See `oracle_registry` in `run.py`  | mmseedbugbasic | No | Oracle type |
| CT | **float** | 2.0 | No  | Coverage threshold |
| FMB | **int** | 25 | No | Fuzz mutation budget |
| C | raw, abs | raw | No | Coverage type |

Upon completion of the fuzzing campaign, a pool of states -on which the policy to be tested-, together with some additional data, is saved in a pickle file in the specified log folder. If that file already exists, fuzzing is skipped and instead, it is loaded. Another pickle file that contains bug information is created after oracle execution in the same folder.

For an example complete execution, first reduce fuzzing budget in `config.py` (e.g. `FUZZ_BUDGET=60`, for one minute of fuzzing) and run the following command:

`python3 run.py -E lunar -R 123 -A policies/lunar_org -CT 0.6 -FMB 25`

This command starts fuzzing on LunarLander environment and upon completion of fuzzing, testing phase starts with default oracle selection. You can check 'pickle'd outputs under `pifuzz_logs` folder or check log file in the same directory to see the number of states in the pool or number of bugs identified.

In the following we provide brief descriptions of important files and folders in the repository:
- `Fuzzer.py`: It includes main fuzzing cycle. It first draws a state (either from the existing pool or from initial state distirbution) and mutates it by calling `Mutator.py`. Later, mutated state is added to the pool if it is diverse from the existing ones in the pool.
- `Oracle.py`: Contains implementations of various oracles. See the paper for more information on oracles.
- `Mutator.py`: It contains several instantiations of mutators. It either mutates a state by taking a number of steps (used in fuzzer), or by relaxing or unrelaxing it (used in oracle).
- `EnvWrapper.py`: It is a wrapper used to interact with environments such as creating environment instances, setting the environment to a particualr state ot taking steps.
- `Seed.py`: Contains seed object.
- `Scheduler.py`: Contains several schedular instantiations used in fuzzer.
- `mod_gym`: Our own fork of OpenAI's [Gym](https://github.com/openai/gym) (forked commit hash: `ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7`). We made changes in `./gym/envs/box2d/lunar_lander.py` and `./gym/envs/box2d/bipedal_walker.py` to accommodate getting and setting states.
- `mod_stable_baselines`: Our own fork of [stable_baselines3](https://stable-baselines3.readthedocs.io/en/master/) (forked commit hash: `503425932f5dc59880f854c4f0db3255a3aa8c1e`). We created this fork to enable training agents in modified Gym environments.

## Testing Your Own Policy

π-fuzz implementation is designed to be modular to allow changes to test policies trained on other benchmarks. For this, one has to extend the wrapper functions such as creating environment, taking a(n) environment/policy step in `EnvWrapper.py`. Other than that, given a state, its relaxation and unrelaxations have to be implemented in `Mutator.py`. Appropriate relaxation and unrelaxation operations depends on the benchmark's nature.

## Replicating Results

Refer to this [link](https://hub.docker.com/repository/docker/practicalformalmethods/pi-fuzz) to download a ready-to-use Docker image. This image contains π-fuzz repository under `/home` directory. We also set up a conda virtual environment that includes all required dependencies to run π-fuzz. One can activate it with the following command `conda activate pifuzz`. The parameters used in the experiments are clearly stated in the paper. We use random seeds from 42 to 49 in repeated experiments. For instance, the following command has to be executed with these random seeds to obtain all **MMSeedBugBasic** results presented in the paper:

`python run.py -E lunar -R 42 -A policies/lunar_org -F inc -O mmseedbugbasic -CT 0.65 -IP 0.2 -FMB 25`

Running this command will take above 24 hours, as there is 24 hours fuzzing campaign in the beginning. For the next oracle type (e.g. `mmbug`) it would take shorter as fuzzing step is not repeated, instead existing data is loaded. Experiment results are saved in specified folder. One can use functions provided in `utils.py` to obtain plots and exact numbers in the tables. Names of the functions are self explanatory.

This image also contains all the data collected in our experiments in a zipped file named `pifuzz_exp.zip`. 
