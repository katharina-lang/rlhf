<p align="center" style="margin-top: -20px;"> 
  <img src="readme_images/Brain Tech (3).png" alt="Projekt-Logo" width="700">
</p>

## Overview

Our project is a fast and modular reimplementation of [Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741), designed for scalable and efficient research in Reinforcement Learning from Human Feedback (RLHF). 
By integrating both synthetic and human feedback for reward modeling, our framework enables systematic exploration of preference-based reinforcement learning in a flexible and extensible setup.



| ![](readme_images/HU-ezgif.com-crop.gif) | ![](readme_images/Spider-ezgif.com-crop.gif) | ![](readme_images/Cheetah-ezgif.com-crop.gif) |
|----------------------------|-------------------------------|-------------------------------|



## Quick Setup

To replicate experiments, clone the repository and install dependencies.
Use a virtal environment and the python version `3.10`

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run experiments
To run experiments
```bash
python -m rlhf.scripts.main
```
This would start an experiment for the `HalfCheeta-v5` environment.
You can set flags to change the environment id and different parameters.
A command with flasg could look like the following:
```bash
python -m rlhf.scripts.main --num-queries 750 --no-synthetic
```
All possible flags to set are in the [arguments](./rlhf/configs/arguments.py) file.

## Results
Comparison of a few environments can be seen here:

## Learn more
For a deep dive into our code, please see the file [walkthrough](./walkthrough.md)
