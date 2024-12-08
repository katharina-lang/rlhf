import numpy as np
import random
import time
import torch
import tyro
from rlhf.configs.arguments import Args
from rlhf.core.ppo import PPO

if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Initialize PPO with reward model
    ppo = PPO(run_name, args, reward_model=True)

    # Start rollout loop
    for iteration in range(1, args.num_iterations + 1):
        # Anneal learning rate
        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

    # Collect rollout data
    ppo.collect_rollout_data()

    # Process trajectories
    print("Processing trajectories and labeling...")
    sequence_length = 60
    num_pairs = 10

    # Create matrices from trajectories
    flattened_buffers = [item for buffer in ppo.trajectory_buffers for item in buffer]
    matrices = ppo.create_matrices_from_trajectories(flattened_buffers, sequence_length)

    # Generate labeled data
    rewards = [[entry["reward"] for entry in buffer] for buffer in ppo.trajectory_buffers]
    flattened_rewards = [reward for sublist in rewards for reward in sublist]
    labeled_data = ppo.compare_and_label_sequences(matrices, flattened_rewards, num_pairs)

    # Save labeled data for later use
    ppo.save_labeled_data(labeled_data, filename=f"labeled_trajectories_{iteration}.npz")

    # Attach labeled data to PPO for reward model training
    ppo.labeled_data = labeled_data

    # Debugging: Print labeled data
    print(f"Labeled data for iteration {iteration}: {ppo.labeled_data}")

    # Train reward model
    ppo.train_reward_model()

    # Calculate advantages and optimize PPO agent
    ppo.advantage_calculation()
    ppo.optimize_agent_and_critic()

    # Logging
    y_pred, y_true = ppo.b_values.cpu().numpy(), ppo.b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    ppo.record_rewards_for_plotting_purposes(explained_var)

# Save the model
if args.save_model:
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    torch.save(ppo.agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

# Clean up
ppo.envs.close()
ppo.writer.close()