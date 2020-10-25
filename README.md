# HIDIL

**Offline Imitation Learning with a Misspecified Simulator**

This repository is code for the paper

> Shengyi Jiang, Jing-Cheng Pang, Yang Yu. **Offline imitation learning with a misspecified simulator**. In: Advances in Neural Information Processing Systems 33 (NeurIPS'20), Virtual Conference, 2020.

## Requirement

- python 3.6
- Anaconda
- gym
- tmux
- tqdm
- tmuxp
- numpy
- pyyaml
- mujoco_py
- matplotlib
- tensorflow==1.13.1

# Instructions

##### For automatic running this code in parallel, you may firstly install tmux and Anaconda. You have access to pretrain expert models on different dynamics mismatch from [Google Drive](https://drive.google.com/drive/folders/1276_TJfClnhz3rDaLkUTsQML1pqRK3-O?usp=sharing).

```bash
# Preparations:
sudo apt-get install tmux
conda create -n your_venv python=3.6
source activate your_venv
pip install -r requirements.txt

# For a minimum running, you can run the following command: 
python main.py --process_num 5 --env_list "Walker2d-v2" --variety_list "0.5" --transfer_type "gravity"

# For customized running, you can modify generate_tmux_config.py and run:
python generate_tmux_config.py --conda_name your_venv
tmuxp load run_all.yaml
```

After one experiment has been done, the summary data will be saved in ./result/summary, which can be loaded by:

```python
import numpy as np
summary = np.load("file_name.npy", allow_pickle=True).item()
```

where summary is a basic type [dict].
