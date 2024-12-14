import torch.nn as nn


# input dim is concatenated state, action
# how Do i want to concatenate? a segment are multiple state action pairs
# each row is a concatenated obs, action
# observation, action, observation, action ?
# for now, segments are 60 obs, action pairs
# matrix with 60 rows
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
