import torch
import torch.nn as nn
import random


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


def train_reward_model_ensemble(
    reward_models, reward_optimizers, labeled_data, device, batch_size=64
):
    """
    Train a list (ensemble) of reward models with mini-batches.
    Each model has a separate loss and optimizer, and each model
    sees the same batches but in a different random order.

    Parameters:
        reward_models: List of RewardModel instances
        reward_optimizers: List of corresponding torch.optim Optimizers
        labeled_data: A list of labeled pairs, where each element has:
            (
                segment_obs_actionOne,
                segment_obs_actionTwo,
                (labelOne, labelTwo),
                (predicted_rewardOne, predicted_rewardTwo),
            )
        device: 'cpu' or 'cuda'
        batch_size: How many pairs per batch (default 64)
    """

    batches = []
    for start in range(0, len(labeled_data), batch_size):
        batches.append(labeled_data[start : start + batch_size])

    for model_index, (model, opt) in enumerate(zip(reward_models, reward_optimizers)):
        total_loss_for_model = 0.0

        batch_indices = list(range(len(batches)))
        random.shuffle(batch_indices)

        for batch_idx in batch_indices:
            batch = batches[batch_idx]

            opt.zero_grad()

            batch_loss = 0.0

            for labeled_pair in batch:
                (
                    segment_obs_actionOne,
                    segment_obs_actionTwo,
                    (labelOne, labelTwo),
                    (predicted_rewardOne, predicted_rewardTwo),
                ) = labeled_pair

                segment_obs_actionOne = torch.tensor(
                    segment_obs_actionOne, device=device
                )
                segment_obs_actionTwo = torch.tensor(
                    segment_obs_actionTwo, device=device
                )

                pred_r1 = model(segment_obs_actionOne).sum()
                pred_r2 = model(segment_obs_actionTwo).sum()

                assert not torch.isnan(pred_r1).any(), "pred_r1 contains NaN values!"
                assert not torch.isinf(pred_r1).any(), "pred_r1 contains Inf values!"
                assert not torch.isnan(pred_r2).any(), "pred_r2 contains NaN values!"
                assert not torch.isinf(pred_r2).any(), "pred_r2 contains Inf values!"
                assert pred_r1.abs().max() < 1e6, "pred_r1 has extreme values!"
                assert pred_r2.abs().max() < 1e6, "pred_r2 has extreme values!"

                # Probability that segment_obs_actionOne is "better"
                prob_one = torch.exp(pred_r1) / (
                    torch.exp(pred_r1) + torch.exp(pred_r2)
                )
                prob_two = 1 - prob_one

                assert not torch.isnan(prob_one).any(), "prob_one contains NaN values!"
                assert not torch.isinf(prob_one).any(), "prob_one contains Inf values!"
                assert not torch.isnan(prob_two).any(), "prob_two contains NaN values!"
                assert not torch.isinf(prob_two).any(), "prob_two contains Inf values!"

                # True labels
                labels = torch.tensor(
                    [labelOne, labelTwo], dtype=torch.float32, device=device
                )

                # Cross-entropy style loss for pairwise preference
                pair_loss = -(
                    labels[0] * torch.log(prob_one + 1e-8)
                    + labels[1] * torch.log(prob_two + 1e-8)
                )

                # Accumulate
                batch_loss += pair_loss

            # Compute gradients batch
            batch_loss.backward()

            # optimizer step
            opt.step()

            # Keep track of the numeric value
            total_loss_for_model += batch_loss.item()

        # print(f"Model {model_index} updated. Total loss: {total_loss_for_model:.4f}")

    print("Reward Models updated")
