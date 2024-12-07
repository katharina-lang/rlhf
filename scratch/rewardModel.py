import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.model(obs)
