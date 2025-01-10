import torch
import torch.nn as nn


# input dim is concatenated state, action
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


# class RewardModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64):
#         super(RewardModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         return self.model(x)


def train_reward_model(reward_model, reward_optimizer, labeled_data, device):
    """Train the reward model not using stored predicted rewards. :("""
    reward_model.train()
    optimizer = reward_optimizer

    optimizer.zero_grad()
    total_loss = 0

    for labeled_pair in labeled_data:
        (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (pred_r1, pred_r2),
        ) = labeled_pair

        segment_obs_actionOne = torch.tensor(segment_obs_actionOne, device=device)
        segment_obs_actionTwo = torch.tensor(segment_obs_actionTwo, device=device)

        pred_r1 = reward_model(segment_obs_actionOne).sum()
        pred_r2 = reward_model(segment_obs_actionTwo).sum()
        labels = torch.tensor(
            [labelOne, labelTwo],
            dtype=torch.float32,
            device=device,
        )

        prob_one = torch.exp(pred_r1) / (torch.exp(pred_r1) + torch.exp(pred_r2))

        prob_two = 1 - prob_one

        pair_loss = -(
            labels[0] * torch.log(prob_one + 1e-8)
            + labels[1] * torch.log(prob_two + 1e-8)
        )
        total_loss += pair_loss

    total_loss.backward()
    optimizer.step()

    print("Reward Model updated")


def train_reward_model_ensemble(reward_models, reward_optimizers, labeled_data, device):
    """
    Train a list (ensemble) of reward models.
    Each model has a separate loss and optimizer.
    """
    model_losses = [0.0 for _ in range(len(reward_models))]
    
    for opt in reward_optimizers:
        opt.zero_grad()

    for labeled_pair in labeled_data:
        (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (predicted_rewardOne, predicted_rewardTwo),
        ) = labeled_pair

        segment_obs_actionOne = torch.tensor(segment_obs_actionOne, device=device)
        segment_obs_actionTwo = torch.tensor(segment_obs_actionTwo, device=device)

        for i, model in enumerate(reward_models):
            pred_r1 = model(segment_obs_actionOne).sum()
            pred_r2 = model(segment_obs_actionTwo).sum()

            prob_one = torch.exp(pred_r1) / (torch.exp(pred_r1) + torch.exp(pred_r2))
            prob_two = 1 - prob_one

            labels = torch.tensor(
                [labelOne, labelTwo], dtype=torch.float32, device=device
            )

            pair_loss = -(
                labels[0] * torch.log(prob_one + 1e-8)
                + labels[1] * torch.log(prob_two + 1e-8)
            )

            model_losses[i] += pair_loss

    for model_loss, opt in zip(model_losses, reward_optimizers):
        model_loss.backward()
        opt.step()

    print("All Reward Models updated (separately).")
