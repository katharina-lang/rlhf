import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# input dim is the dimension of the state (or concatenated state and action)
# not working yet prototype
class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
