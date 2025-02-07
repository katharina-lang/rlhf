<p align="center" style="margin-top: -20px;"> 
  <img src="documents/Brain Tech (3).png" alt="Projekt-Logo" width="700">
</p>

## Overview

Our project is a fast and modular reimplementation of [Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741), designed for scalable and efficient research in Reinforcement Learning from Human Feedback (RLHF). 
By integrating both synthetic and human feedback for reward modeling, our framework enables systematic exploration of preference-based reinforcement learning in a flexible and extensible setup.



| ![](documents/HU-ezgif.com-crop.gif) | ![](documents/Walker.gif) | ![](documents/DesignohneTitel2-ezgif.com-crop.gif) |
|----------------------------|-------------------------------|-------------------------------|



## Quick Setup

To replicate experiments, clone the repository and install dependencies.
Use a virtual environment and the python version `3.10`

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run experiments
To run experiments either use (or create) the shell scripts in the [scripts](./scripts/) directory (you might need to set the executable permission):
```bash
chmod +x scripts/halfc.sh
scripts/halfc.sh
```

or run the file directly

```bash
python -m rlhf.main
```

These would start an experiment for the `HalfCheeta-v5` environment.
You can set flags to change the environment id and different parameters.
A command with flags could look like the following:
```bash
python -m rlhf.main --num-queries 750 --no-synthetic
```
All possible flags can be found in the [arguments](./rlhf/configs/arguments.py) file.

If the flag `--no-synthetic` is set, human labeling is required. The system will launch a **Flask** web application where the user labels preferences through an interface in **Google Chrome**.
To access the labeling interface, open **Google Chrome** and navigate to the address shown in the terminal.

## Results
An example result can be seen here (Humanoid Standup with all CleanRl default parameters):

<img src="./documents/stats/episodicRHs.png" alt="Humanoid Standup - Episodic RHs" width="350" height="350">

## Learn more
For a deep dive into our code, please see the file [walkthrough](./walkthrough.md).
