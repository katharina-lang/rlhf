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


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, pairwise_data):
        self.data = pairwise_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seg1, seg2, preference = self.data[idx]
        state1 = torch.tensor(
            np.array([step["state"] for step in seg1]), dtype=torch.float32
        )
        state2 = torch.tensor(
            np.array([step["state"] for step in seg2]), dtype=torch.float32
        )
        return state1, state2, preference


def reward_model_loss(reward1, reward2, preference):
    prob = torch.sigmoid(reward1 - reward2)
    return -torch.mean(
        preference * torch.log(prob) + (1 - preference) * torch.log(1 - prob)
    )
