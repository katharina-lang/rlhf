# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import time
import torch
import numpy as np
import tyro
from configs.arguments import Args
from core.ppo import PPO


if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    ppo = PPO(run_name, args, reward_model=True)

    # starts rollout loop
    for iteration in range(1, args.num_iterations + 1):
        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

        ppo.collect_rollout_data()

        # trajectories
        # habe self.preference_database
        # jetzt nehme ich zufällige segmente raus, label sie
        # speichere sie in labeled data, die labels müssen mit den trajektorien gespeichert werden
        # damit trainiere ich dann

        ppo.train_reward_model()
        ppo.advantage_calculation()
        ppo.optimize_agent_and_critic()

        y_pred, y_true = ppo.b_values.cpu().numpy(), ppo.b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ppo.record_rewards_for_plotting_purposes(explained_var)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(ppo.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    ppo.envs.close()
    ppo.writer.close()