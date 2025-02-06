<p align="center" style="margin-top: -20px;"> 
  <img src="Brain Tech (3).png" alt="Projekt-Logo" width="700">
</p>

## Overview

This project is a reimplementation of the paper "Deep Reinforcement Learning from Human Preferences" (Christiano et al., 2017). Our goal is to replicate and explore Reinforcement Learning from Human Feedback (RLHF) using a modularized training setup and incorporating synthetic and human feedback for reward modeling.



https://github.com/user-attachments/assets/56a07f7e-580f-49e6-b700-45bcb90dda22


https://github.com/user-attachments/assets/17e34f8b-6143-4d8e-b684-129e7fff1f61



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

## Code Walkthrough
A code walkthrough is provided in the file [walkthrough](./walkthrough.md)
