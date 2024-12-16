import time
import torch
import numpy as np
import tyro
from rlhf.configs.arguments import Args
from rlhf.core.ppo import PPO


if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    ppo = PPO(run_name, args, reward_model=True)

    # Start the rollout loop
    start_rollout_loop(ppo, args.num_iterations)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(ppo.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    ppo.envs.close()
    ppo.writer.close()
    

def start_rollout_loop(ppo, num_iterations):
    """
    Starts the main rollout loop for training the agent and the reward model.
    
    Parameters:
        ppo (PPO): The PPO instance managing the agent and reward model training.
        num_iterations (int): Number of iterations to run the rollout loop.
    """
    for iteration in range(1, num_iterations + 1):
        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * ppo.args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

        ppo.collect_rollout_data()

        # Process and train the reward model
        ppo.train_reward_model()

        # Perform advantage calculation and optimization
        ppo.advantage_calculation()
        ppo.optimize_agent_and_critic()

        # Calculate explained variance for debugging purposes
        y_pred, y_true = ppo.b_values.cpu().numpy(), ppo.b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ppo.record_rewards_for_plotting_purposes(explained_var)


