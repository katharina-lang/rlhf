# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

# python -m src.ppo_continuous_action --capture-video --save-model
import os
import random
import time


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from src.reward_model import RewardModel, RewardDataset, reward_model_loss
from src.init_arguments import Args
from src.utils import (
    make_env,
    Agent,
    generate_pairwise_data,
    save_segment_video,
    show_and_get_feedback,
)


def train_reward_model(
    reward_model,
    reward_optimizer,
    pairwise_data,
    global_step,
    epochs=3,
    writer=None,
):
    dataset = RewardDataset(pairwise_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for state1, state2, preference in dataloader:
            state1, state2, preference = (
                state1.to(device),
                state2.to(device),
                preference.to(device),
            )
            reward1 = reward_model(state1.mean(dim=1))
            reward2 = reward_model(state2.mean(dim=1))
            loss = reward_model_loss(reward1, reward2, preference)

            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()

            # Log metrics
            correct = ((reward1 > reward2) == preference).sum().item()
            total = len(preference)
            accuracy = correct / total
            writer.add_scalar("reward_model/loss", loss.item(), global_step)
            writer.add_scalar("reward_model/accuracy", accuracy, global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    ###
    # set up trajectory collection
    #### Trajectory Calc
    num_envs = args.num_envs
    trajectory_buffers = [[] for _ in range(num_envs)]
    segment_size = 90
    trajectory_segments = []
    pairwise_data = []
    pairwise_generation_interval = 100
    step_counter = 0
    all_segments_for_feedback = []

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = "cpu"

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize the reward model
    reward_model = RewardModel(
        input_dim=np.prod(envs.single_observation_space.shape)
    ).to(device)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-4)

    # Load the policy model if it exists
    os.makedirs("models", exist_ok=True)
    models_agent = [f for f in os.listdir("models") if "agent" in f]
    models_agent_sorted = sorted(
        models_agent,
        key=lambda x: int(x.split("__")[-1].split(".")[0].split("_")[0]),
        reverse=True,
    )
    if models_agent_sorted and args.exp_name in models_agent_sorted[0]:
        model_path = os.path.join("models", models_agent_sorted[0])
        agent.load_state_dict(torch.load(model_path))
        agent.eval()  # Set the model to evaluation mode
        print(f"Loaded model from {model_path}")
    else:
        print("No saved agent model found. Training from scratch.")

    models_reward = [f for f in os.listdir("models") if "reward" in f]
    models_reward_sorted = sorted(
        models_reward,
        key=lambda x: int(x.split("__")[-1].split(".")[0].split("_")[0]),
        reverse=True,
    )
    if models_reward_sorted and args.exp_name in models_reward_sorted[0]:
        model_path = os.path.join("models", models_reward_sorted[0])
        reward_model.load_state_dict(torch.load(model_path))
        reward_model.eval()  # Set the model to evaluation mode
        print(f"Loaded model from {model_path}")
    else:
        print("No saved reward model found. Training from scratch.")

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            step_counter += args.num_envs  ###

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )

            if not args.original:
                with torch.no_grad():
                    predicted_reward = reward_model(
                        torch.tensor(next_obs, dtype=torch.float32).to(device)
                    )
                rewards[step] = predicted_reward.view(-1)

            next_done = np.logical_or(terminations, truncations)
            if args.original:
                rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            # # # Collect trajectory data for each environment
            #### Trajectory Calc
            if not args.original:
                for env_idx in range(num_envs):
                    trajectory_buffers[env_idx].append(
                        {
                            "state": next_obs[env_idx].cpu().numpy(),
                            "action": action[env_idx].cpu().numpy(),
                            "reward": reward[env_idx],
                        }
                    )

                    if len(trajectory_buffers[env_idx]) == segment_size:
                        segment = trajectory_buffers[env_idx][:segment_size]
                        trajectory_buffers[env_idx] = trajectory_buffers[env_idx][
                            segment_size:
                        ]
                        trajectory_segments.append(segment)

                    if terminations[env_idx] or truncations[env_idx]:
                        # If the buffer contains fewer steps, discard it (to ensure fixed-size segments)
                        trajectory_buffers[env_idx] = []

                if step_counter >= pairwise_generation_interval or any(terminations):
                    step_counter = 0

                    trajectory_segments = [
                        segment
                        for segment in trajectory_segments
                        if len(segment) == segment_size
                    ]

                    if args.use_human_feedback:
                        all_segments_for_feedback.extend(trajectory_segments)
                        while len(all_segments_for_feedback) > 1:
                            segment1 = all_segments_for_feedback.pop(0)
                            segment2 = all_segments_for_feedback.pop(0)
                            feedback = show_and_get_feedback(
                                segment1, segment2, args.env_id
                            )

                            if feedback:
                                pairwise_data.append(feedback)
                    else:
                        if len(trajectory_segments) > 1:
                            pairwise_data.extend(
                                generate_pairwise_data(trajectory_segments)
                            )

                    # filter out if segments are not the same size
                    if len(trajectory_segments) > 1:
                        pairwise_data.extend(
                            generate_pairwise_data(trajectory_segments)
                        )

                # Train reward model after collecting enough pairwise data
                if len(pairwise_data) >= 10:
                    # Reward model training dataset
                    reward_dataset = RewardDataset(pairwise_data)
                    reward_dataloader = torch.utils.data.DataLoader(
                        reward_dataset, batch_size=32, shuffle=True
                    )
                    ###
                    train_reward_model(
                        reward_model,
                        reward_optimizer,
                        pairwise_data,
                        global_step,
                        epochs=3,
                        writer=writer,
                    )
                    pairwise_data.clear()

                # # # collect ending

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    # ### human labeler
    # pairwise_data = []
    # for _ in range(num_pairs):  # Generate pairs for human feedback
    #     seg1, seg2 = random.sample(trajectory_segments, 2)
    #     # Ask a human to label the preference
    #     print("Segment 1:", seg1)
    #     print("Segment 2:", seg2)
    #     preference = int(input("Which segment is better? (1 for seg1, 0 for seg2): "))
    #     pairwise_data.append((seg1, seg2, preference))

    if args.save_model:
        # save the trained model
        torch.save(agent.state_dict(), f"models/{run_name}_agent.pt")
        print(f"Model saved to models/{run_name}_agent.pt")
        torch.save(reward_model.state_dict(), f"models/{run_name}_reward_model.pt")
        print(f"Model saved to models/{run_name}_reward_model.pt")

    envs.close()
    writer.close()
