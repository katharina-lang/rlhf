import torch
import torch.nn as nn
import random


# input dim is concatenated state, action
class RewardModel(nn.Module):
    """
    Neural network-based reward model for predicting reward values.

    Args:
        input_dim (int): The input dimension, consisting of concatenated state and action features.
        hidden_dim (int, optional): The number of hidden units in the model. Defaults to 64.
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.3.
    """

    def __init__(self, input_dim, hidden_dim=64, dropout_p=0.3):
        super(RewardModel, self).__init__()
        self.dropout_p = dropout_p
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            self.dropout1,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout2,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward pass through the reward model.

        Args:
            x (torch.Tensor): Input tensor representing the concatenated state and action.

        Returns:
            torch.Tensor: Predicted reward value.
        """
        return self.model(x)

    def set_dropout(self, new_p):
        """
        Updates the dropout probability.

        Args:
            new_p (float): New dropout probability value.
        """
        self.dropout_p = new_p
        self.dropout1.p = new_p
        self.dropout2.p = new_p


def train_reward_model_ensemble(
    reward_models,
    reward_optimizers,
    labeled_data,
    val_data,
    device,
    batch_size=64,
    writer=None,
    global_step=0,
    anneal_dropout=False,
    default_dropout=0.3,
):
    """
    Trains an ensemble of reward models using labeled trajectory segments.

    Args:
        reward_models (list): List of `RewardModel` instances.
        reward_optimizers (list): List of corresponding `torch.optim.Optimizer` instances.
        labeled_data (list): List of labeled pairs containing:
                             (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo)).
        val_data (list): List of validation data pairs (same structure as labeled_data).
        device (str): The computation device, either "cpu" or "cuda".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging.
        global_step (int, optional): Global step counter for logging purposes. Defaults to 0.
        anneal_dropout (bool, optional): Whether to adjust dropout based on validation loss. Defaults to False.
        default_dropout (float, optional): Default dropout probability. Defaults to 0.3.
    """

    train_pairs = labeled_data

    random.shuffle(train_pairs)

    train_batches = []
    for start in range(0, len(train_pairs), batch_size):
        train_batches.append(train_pairs[start : start + batch_size])

    for model_idx, (model, opt) in enumerate(zip(reward_models, reward_optimizers)):
        total_train_loss = 0.0

        # Shuffle the order of batches for each model
        batch_indices = list(range(len(train_batches)))
        random.shuffle(batch_indices)

        for b_idx in batch_indices:
            batch = train_batches[b_idx]

            opt.zero_grad()
            batch_loss = 0.0

            for labeled_pair in batch:
                (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo)) = (
                    labeled_pair
                )

                segment_obs_actionOne = torch.tensor(
                    segment_obs_actionOne, device=device
                )
                segment_obs_actionTwo = torch.tensor(
                    segment_obs_actionTwo, device=device
                )

                # Forward pass
                pred_r1 = model(segment_obs_actionOne).sum()
                pred_r2 = model(segment_obs_actionTwo).sum()

                # Probability that segment_obs_actionOne is "better"
                prob_one = torch.exp(pred_r1) / (
                    torch.exp(pred_r1) + torch.exp(pred_r2)
                )
                prob_two = 1 - prob_one

                labels = torch.tensor(
                    [labelOne, labelTwo], dtype=torch.float32, device=device
                )

                cross_val_loss = -(
                    labels[0] * torch.log(prob_one + 1e-8)
                    + labels[1] * torch.log(prob_two + 1e-8)
                )

                batch_loss += cross_val_loss

            # Backprop on the batch
            batch_loss.backward()
            opt.step()

            total_train_loss += batch_loss.item()

        if val_data:

            random.shuffle(val_data)
            val_loss = compute_reward_model_loss(
                model, val_data, device, batch_size=batch_size
            )

            if (
                anneal_dropout
                and (val_loss / len(val_data) / (total_train_loss / len(train_pairs)))
                > 1.8
            ):
                model.set_dropout(min(model.dropout_p + 0.05, 0.5))
            elif anneal_dropout:
                model.set_dropout(default_dropout)

            if writer is not None and val_data:
                # Log the *(avg)* training loss, for the global timestep
                writer.add_scalar(
                    f"Model_{model_idx}/TrainLoss",
                    total_train_loss / len(train_pairs),
                    global_step,
                )
                writer.add_scalar(
                    f"Model_{model_idx}/ValLoss",
                    val_loss / len(val_data),
                    global_step,
                )

                print(
                    f"Model {model_idx} "
                    f"=> Train Loss: {total_train_loss/len(train_pairs):.4f}, Val Loss: {val_loss/len(val_data):.4f}"
                )

    print("All Reward Models updated.")


def compute_reward_model_loss(model, data_pairs, device, batch_size=64):
    """
    Computes the total loss for a given reward model on labeled trajectory pairs.

    Args:
        model (RewardModel): The reward model to evaluate.
        data_pairs (list): List of labeled trajectory pairs:
                           (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo)).
        device (str): The computation device, either "cpu" or "cuda".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.

    Returns:
        float: The total computed loss over all data pairs.
    """

    model.eval()  # eval mode (disables dropout, etc.)

    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, len(data_pairs), batch_size):
            batch = data_pairs[start : start + batch_size]
            batch_loss = 0.0

            for labeled_pair in batch:
                (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo)) = (
                    labeled_pair
                )

                segment_obs_actionOne = torch.tensor(
                    segment_obs_actionOne, device=device
                )
                segment_obs_actionTwo = torch.tensor(
                    segment_obs_actionTwo, device=device
                )

                pred_r1 = model(segment_obs_actionOne).sum()
                pred_r2 = model(segment_obs_actionTwo).sum()

                prob_one = torch.exp(pred_r1) / (
                    torch.exp(pred_r1) + torch.exp(pred_r2)
                )
                prob_two = 1 - prob_one

                labels = torch.tensor(
                    [labelOne, labelTwo], dtype=torch.float32, device=device
                )

                pair_loss = -(
                    labels[0] * torch.log(prob_one + 1e-8)
                    + labels[1] * torch.log(prob_two + 1e-8)
                )

                batch_loss += pair_loss.item()

            total_loss += batch_loss

    model.train()
    return total_loss
