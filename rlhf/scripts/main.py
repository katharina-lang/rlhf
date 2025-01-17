import time
import torch
import numpy as np
import tyro
from rlhf.configs.arguments import Args
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import train_reward_model_ensemble
from rlhf.core.labeling import Labeling


def start_rollout_loop(ppo, num_iterations):
    """
    Starts the main rollout loop for training the agent and the reward model.

    Parameters:
        ppo (PPO): The PPO instance managing the agent and reward model training.
        num_iterations (int): Number of iterations to run the rollout loop.
    """

    segment_size = 60

    total_queries = ppo.args.num_queries
    min_queries_per_training = 5
    amount_of_trainings = total_queries // min_queries_per_training
    div = num_iterations // amount_of_trainings

    queries_trained = 0

    updates = num_iterations

    for iteration in range(1, num_iterations + 1):
        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * ppo.args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

        ppo.collect_rollout_data()

        if iteration % div == 0:
            queries = min(min_queries_per_training, total_queries)

            if queries > 0:
                total_queries -= min_queries_per_training
                queries_trained += queries

                labeling = Labeling(segment_size)
                labeled_data = labeling.get_labeled_data(
                    ppo.obs_action_pair_buffer,
                    ppo.env_reward_buffer,
                    ppo.predicted_rewards_buffer,
                    queries,
                )

                train_reward_model_ensemble(
                    ppo.reward_models, ppo.optimizers, labeled_data, ppo.device
                )

        ppo.advantage_calculation()

        ppo.agent.optimize_agent_and_critic(
            ppo.obs,
            ppo.actions,
            ppo.logprobs,
            ppo.advantages,
            ppo.returns,
            ppo.values,
            ppo.optimizer,
            ppo.args,
        )

        # Calculate explained variance for debugging purposes
        y_pred, y_true = ppo.values.cpu().numpy(), ppo.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ppo.record_rewards_for_plotting_purposes(explained_var)

    print(queries_trained)


if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    ppo = PPO(run_name, args)
    # Start the rollout loop
    start_rollout_loop(ppo, args.num_iterations)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(ppo.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    ppo.envs.close()
    ppo.writer.close()
