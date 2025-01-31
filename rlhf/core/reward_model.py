import torch
import torch.nn as nn
import random


# input dim is concatenated state, action
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_p=0.3):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


def compute_reward_model_loss(model, data_pairs, device, batch_size=64):
    """
    Computes total (summed) loss of the given model on 'data_pairs'.
    data_pairs is a flat list of:
        (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo), _).

    This function does NOT update the model.
    """

    model.eval()  # eval mode (disables dropout, etc.)

    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, len(data_pairs), batch_size):
            batch = data_pairs[start : start + batch_size]
            batch_loss = 0.0

            for labeled_pair in batch:
                (
                    segment_obs_actionOne,
                    segment_obs_actionTwo,
                    (labelOne, labelTwo),
                    _,
                ) = labeled_pair


                segment_obs_actionOne = torch.tensor(
                    segment_obs_actionOne, device=device
                )
                segment_obs_actionTwo = torch.tensor(
                    segment_obs_actionTwo, device=device
                )

                # Forward pass
                pred_r1 = model(segment_obs_actionOne).sum()
                pred_r2 = model(segment_obs_actionTwo).sum()

                # Probability that segment_obs_actionOne is better

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


def train_reward_model_ensemble(
    reward_models,
    reward_optimizers,
    labeled_data,
    val_data,
    device,
    batch_size=64,
    writer=None,
    global_step=0,
    epochs=1,
):
    """
    Train a list (ensemble) of reward models with mini-batches.
    Each model has a separate loss and optimizer.
    We do an 80/20 split on the labeled pairs for train vs. validation.

    reward_models:      list of RewardModel instances
    reward_optimizers:  list of corresponding torch.optim.Optimizer
    labeled_data:       list of labeled pairs of the form:
                                  (segment_obs_actionOne,
                                   segment_obs_actionTwo,
                                   (labelOne, labelTwo),
                                   (predicted_rewardOne, predicted_rewardTwo))
    device:             "cpu" or "cuda"
    batch_size:         number of pairs in each batch
    epochs:             how many epochs to train for
    writer:             optional, a tensorboard SummaryWriter for logging
    """

    train_pairs = labeled_data

    for epoch in range(epochs):
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
                    (
                        segment_obs_actionOne,
                        segment_obs_actionTwo,
                        (labelOne, labelTwo),
                        _,
                    ) = labeled_pair

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

                    # Accumulate loss for this mini-batch
                    batch_loss += cross_val_loss

                # Backprop on the batch
                batch_loss.backward()
                opt.step()

                total_train_loss += batch_loss.item()

            if val_data:
                # validation loss verhältnis zu normalen loss 1.1 bis 1.5
                # dropout während training verändern?

                random.shuffle(val_data)
                val_loss = compute_reward_model_loss(
                    model, val_data, device, batch_size=batch_size
                )

                # Log the training and validation losses
                if writer is not None:
                    # Log the *(avg)total* training loss, for the global timestep
                    # epochs always = 1
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
                    f"[Epoch {epoch+1}/{epochs}] Model {model_idx} "
                    f"=> Train Loss: {total_train_loss/len(train_pairs):.4f}, Val Loss: {val_loss/len(val_data):.4f}"
                )

    print("All Reward Models updated.")
