
# SAIL: A Stable and Anthropomorphic Intelligent Locomotion Method for Humanoid Robots in Complex Terrains via Style Evaluation #

## System Requirements

- **Operating System**: Recommended Ubuntu 20.04 or later  
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 525 or later  

---

## Installation

```bash
conda create -n SAIL python=3.8
conda activate SAIL
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
cd isaacgym/python && pip install -e .
cd ~/SAIL/rsl_rl && pip install -e .
cd ~/SAIL && pip install -e .
pip install wandb tqdm opencv-python ipdb pyfqmr
```

## Usage

```bash
cd ~/SAIL/legged_gym/scripts
```

### 1. Train
Run the following command to start training:
```bash
python train.py  --task=g1 ---num_envs=6144 --max_iterations=<iteration> --headless
```
Train 10-20k iterations (at least 10k recommended). You can train policy at arbitary gpu `#` as long as you set `--rl_device="cuda:#"`.

### 2. Play
To visualize the training results in Gym, run the following command:
```bash
python play.py --task=g1 --load_run=<date_time>_<run_name> --checkpoint==<iteration> --resume
```

#### Viewer Usage
Can be used in both IsaacGym and web viewer.
- `ALT + Mouse Left + Drag Mouse`: move view.
- `[ ]`: switch to next/prev robot.
- `Space`: pause/unpause.
- `F`: switch between free camera and following camera.

### 3. Export Network
Play exports the Actor network, saving it in logs/{experiment_name}/exported/policies:
Standard networks (MLP) are exported as `policy_1.pt`.

## Arguments
- "--task",  Resume training or start testing from a checkpoint. Overrides config file if provided.
- "--resume", default: False, Resume training from a checkpoint.
- "--experiment_name", Name of the experiment to run or load. Overrides config file if provided.
- "--run_name", Name of the run. Overrides config file if provided.
- "--load_run", Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.
- "--checkpoint", Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.
- "--headless", default: False, Force display off at all times.
- "--rl_device", default: "cuda:0", Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..).
- "--num_envs", Number of environments to create. Overrides config file if provided.
- "--seed", Random seed. Overrides config file if provided.
- "--max_iterations",  Maximum number of training iterations. Overrides config file if provided.
- "--lin_vel_x", Linear velocity in x direction (m/s).


## Acknowledgement

- https://github.com/leggedrobotics/legged_gym
- https://github.com/leggedrobotics/rsl_rl
- https://github.com/chengxuxin/extreme-parkour
- https://github.com/fan-ziqi/rl_sar
- https://github.com/unitreerobotics/unitree_rl_gym
- https://github.com/unitreerobotics/unitree_sdk2_python

## LICENSE
This project is licensed under the [BSD-3-Clause License]. For details, please refer to the LICENSE file.

