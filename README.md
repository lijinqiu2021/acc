# Leveraging Privileged Information for Partially Observable Reinforcement Learning
This repository includes the source code for the paper "Leveraging Privileged Information for Partially Observable Reinforcement Learning".

## Installation
Make sure you have Python 3.7+ installed. Install dependencies:
```
pip3 install -r requirements.txt
```
### Atari
For Atari experiments:
- We use `Envpool` the C++-based high-performance vectorized environment package for Atari games. Note that Envpool does not work in Windows and macOS, please see [https://github.com/sail-sg/envpool](https://github.com/sail-sg/envpool) for more details.
- We use `JAX` the high-performance array computing package instead of PyTorch for training neural networks. If you want to install JAX with NVIDIA GPU support, please see [https://github.com/google/jax](https://github.com/google/jax) for more details.

### MiniGrid
For MiniGrid experiments:
- We use `MiniGrid` the python-based environment package for MiniGrid games. Please see [https://github.com/Farama-Foundation/MiniGrid](https://github.com/Farama-Foundation/MiniGrid) for more details.
- We use `PyTorch` for training neural networks. Please see [https://pytorch.org/](https://pytorch.org/) for more details.

## Training
### Atari
We provide codes for 4 methods, including ACC, AEC, AOC, Dropout ($\beta=1/2$ and $\beta=2/3$). Every detail of a method is placed in a single standalone file: `acc.py`, `aec.py`, `aoc.py`, `dropout.py`. 

For example, run an experiment of ACC in the "Pong" environment: 
```
python3 acc.py \
    --seed 1 \
    --env-id Pong-v5 \
    --total-timesteps 5000000
```

For more customized configuration of training, you can directly obtain the documentation by using the --help flag.
```
python3 acc.py --help
```

### MiniGrid
We provide codes for 4 methods, including ACC, AEC, AOC, Unbiased Asymmetric Actor-Critic. All methods are integrated and can be executed using the algo parameter (`acc`, `aec`, `acc`, `uaac`).

For example, run an experiment of ACC in the "Empty-16x16" environment: 
```
python3 train.py \
    --algo acc \
    --seed 1 \
    --env-id MiniGrid-Empty-16x16-v0 \
    --total-timesteps 1200000
```

For more customized configuration of training, you can directly obtain the documentation by using the --help flag.
```
python3 train.py --help
```

## Visualization
By default, we record all the training metrics (including average returns, etc.) via `Tensorboard` in the runs folder. To visualize them, You can run:
```
tensorboard --logdir runs
```
You can also use `Wandb` to track the experiments by the following command:
```
wandb login # only required for the first time
python3 XXX.py \
    --track \
    --wandb-project-name ACC
```

## Acknowledgment
This code implementation of PPO is largely based on the CleanRL library ([https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)).

This code implementation of A2C is largely based on the RL Starter Files library ([https://github.com/lcswillems/rl-starter-files](https://github.com/lcswillems/rl-starter-files)).
