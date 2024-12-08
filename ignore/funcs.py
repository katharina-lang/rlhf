def collect_pair(self):
    pass


def collect_preferences(self, num_pairs=10):
    """Collect preferences based on real rewards."""
    # human later
    selected_pairs = self.select_trajectories_by_reward(num_pairs)
    self.preference_database.extend(selected_pairs)


def train_reward_model(self):
    """Train reward model from preferences."""
    if not self.preference_database:
        return

    loss_fn = nn.CrossEntropyLoss()
    print(self.preference_database)
    for traj1, traj2, preference in self.preference_database:
        # Prepare trajectory observations
        obs1 = torch.tensor([step["obs"] for step in traj1], dtype=torch.float32).to(
            self.device
        )
        obs2 = torch.tensor([step["obs"] for step in traj2], dtype=torch.float32).to(
            self.device
        )

        # Predict cumulative rewards
        r1 = self.reward_model(obs1).sum()  # Sum of predicted rewards for trajectory 1
        r2 = self.reward_model(obs2).sum()  # Sum of predicted rewards for trajectory 2

        logits = torch.stack([r1, r2], dim=0)
        target = torch.tensor([preference - 1]).to(
            self.device
        )  # Preference target: 0 or 1

        # Compute loss and update the reward model
        loss = loss_fn(logits.unsqueeze(0), target)
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()


def select_trajectories_by_reward(self, num_pairs):
    """Select trajectory pairs based on cumulative rewards."""
    # Compute cumulative rewards for each trajectory
    ranked_trajectories = sorted(
        self.preference_database,
        key=lambda traj: sum([step["reward"] for step in traj]),
        reverse=True,
    )

    # Select pairs: top vs lower-ranked trajectories
    selected_pairs = []
    for i in range(num_pairs):
        if i + 1 < len(ranked_trajectories):
            traj1 = ranked_trajectories[i]
            traj2 = ranked_trajectories[-(i + 1)]  # Pair with a lower-ranked trajectory
            preference = (
                1
                if sum([step["reward"] for step in traj1])
                > sum([step["reward"] for step in traj2])
                else 2
            )
            selected_pairs.append((traj1, traj2, preference))

    return selected_pairs
