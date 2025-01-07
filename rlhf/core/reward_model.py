import torch
import torch.nn as nn


# input dim is concatenated state, action
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


def train_reward_model(reward_model, reward_optimizer, labeled_data, device, epochs=4):
    """
    Trainiert das Reward-Modell anhand der gelabelten Daten.
    """
    reward_model.train()

    for _ in range(epochs):
        epoch_loss = 0

        for labeled_pair in labeled_data:
            (
                segment_obs_actionOne,
                segment_obs_actionTwo,
                (labelOne, labelTwo),
                (predicted_rewardOne, predicted_rewardTwo),
            ) = labeled_pair

            segment_obs_actionOne = torch.tensor(segment_obs_actionOne, device=device)
            segment_obs_actionTwo = torch.tensor(segment_obs_actionTwo, device=device)

            predicted_rewardOne = reward_model(segment_obs_actionOne).sum()
            predicted_rewardTwo = reward_model(segment_obs_actionTwo).sum()
            labels = torch.tensor(
                [labelOne, labelTwo], dtype=torch.float32, device=device
            )

            prob_one = torch.exp(predicted_rewardOne) / (
                torch.exp(predicted_rewardOne) + torch.exp(predicted_rewardTwo)
            )
            prob_two = 1 - prob_one

            loss = -(
                labels[0] * torch.log(prob_one + 1e-8)
                + labels[1] * torch.log(prob_two + 1e-8)
            )

            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()

            epoch_loss += loss.item()

