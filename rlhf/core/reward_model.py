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

            assert not torch.isnan(pred_r1).any(), "pred_r1 contains NaN values!"
            assert not torch.isinf(pred_r1).any(), "pred_r1 contains Inf values!"
            assert not torch.isnan(pred_r2).any(), "pred_r2 contains NaN values!"
            assert not torch.isinf(pred_r2).any(), "pred_r2 contains Inf values!"
            assert pred_r1.abs().max() < 1e6, "pred_r1 has extreme values!"
            assert pred_r2.abs().max() < 1e6, "pred_r2 has extreme values!"

            prob_one = torch.exp(pred_r1) / (torch.exp(pred_r1) + torch.exp(pred_r2))
            prob_two = 1 - prob_one

            assert not torch.isnan(prob_one).any(), "prob_one contains NaN values!"
            assert not torch.isinf(prob_one).any(), "prob_one contains Inf values!"
            assert not torch.isnan(prob_two).any(), "prob_two contains NaN values!"
            assert not torch.isinf(prob_two).any(), "prob_two contains Inf values!"

            labels = torch.tensor(
                [labelOne, labelTwo], dtype=torch.float32, device=device
            )

            pair_loss = -(
                labels[0] * torch.log(prob_one + 1e-8)
                + labels[1] * torch.log(prob_two + 1e-8)
            )

            model_losses[i] = model_losses[i] + pair_loss

    for model_loss, opt in zip(model_losses, reward_optimizers):
        model_loss.backward()
        opt.step()

    print("All Reward Models updated (separately).")
