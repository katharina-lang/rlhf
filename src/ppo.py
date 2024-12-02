import argparse
import os
import random
import time
from distutils.util import strtobool

import wandb
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# to launch tensorboard:
# tensorboard --logdir runs, clock on link
# to train an agent (models get saved):
# python -m src.ppo --total-timesteps 1000000 --num-steps 2048 --num-envs 8 --update-epochs 10 --learning-rate 1e-4 --capture-video


def make_env(gym_id, seed, idx, capture_video, run_name):
    """gym_id is the name of the env, e.g CartPole-v1"""

    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")

        env.reset(seed=seed)  # modern replacement for deprecated env.seed(seed) method
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join("videos", run_name),
                episode_trigger=lambda t: t % 100 == 0,
                disable_logger=True,
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(
        layer.weight, std
    )  # PPO uses orthogonal init on layer weights
    torch.nn.init.constant_(
        layer.bias, bias_const
    )  # PPO uses constant_ init on layers bias
    # pytroch would use different layer initialization methods
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        # critic has three linear layers with the tanh as activation function
        # input shape to the first Linear Layer is the product of the obs space shape
        # std low -> layers parameters will have similar scalar values -> probability of taking each action is similar
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)  # critic inference

    def get_action_and_value(self, x, action=None):
        # When doing actors inference it's best to bundle the results with critics inference
        logits = self.actor(x)  # unnormalized action probabilities
        probs = Categorical(logits=logits)  # basically softmax to get probability

        if action is None:  # rollout phase -> we sample actions
            action = probs.sample()

        # actions, logprobabilities of the actions, entropy of the action probability distribution, values
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="CartPole-v1",
        help="the id of the gym environment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the environment")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=25000,
        help="total timesteps of the experiment",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled , 'torch.backends.cudnn.deterministic=False'",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will not be enabled by default",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="wether to capture videos of the agent performances (check out 'videos' folder)",
    )

    # wandb
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps to run in each environment per policy rollout",
    )
    # how much data we collect per policy rollout 4*128=512
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for üolicy value networks",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use GAE for advantage computation",
    )  # General Advantage Estimation for ppo
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="the K epcohs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles wheter or not to use a clipped loss for the value function as per paper",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshhold",
    )  # 0.015 is the default value in OpenAi spinning up

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 512 // 4 = 128

    return args


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)  # Ensure the directory for saved models exists

    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(
        f"runs/{run_name}"
    )  # to have a logged and readable tensorboard file
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Try not to modify
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available and args.cuda else "cpu")
    device = torch.device("cpu")  # i dont have a gpu

    env_fns = [
        make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Load the model if it exists
    models = [f for f in os.listdir("models")]
    models_sorted = sorted(
        models,
        key=lambda x: int(x.split("__")[-1].split(".")[0].split("_")[0]),
        reverse=True,
    )
    if models_sorted and args.gym_id in models_sorted[0]:
        model_path = os.path.join("models", models_sorted[0])
        agent.load_state_dict(torch.load(model_path))
        agent.eval()  # Set the model to evaluation mode
        print(f"Loaded model from {model_path}")
    else:
        print("No saved model found. Training from scratch.")

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
    global_step = 0  # track the number of environment steps
    start_time = time.time()  # helps to track number of frames per second later
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)  # store initial observation
    next_done = torch.zeros(args.num_envs).to(
        device
    )  # to store termination condition to be false
    num_updates = (
        args.total_timesteps // args.batch_size
    )  # num of iterations/updates for entirety of training

    for update in range(1, num_updates + 1):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate  # learning rate gets lower
            # each update corresponds to one iteration in the training loop
            optimizer.param_groups[0][
                "lr"
            ] = lrnow  # pytorch api to update learning rate

        # policy rollout
        for step in range(0, args.num_steps):
            # because each step happens at the vector environment
            # we implement th global step by the number of envs
            global_step += 1 * args.num_envs
            obs[step] = next_obs  # store next observation
            dones[step] = next_done  # store next done

            # during the rollout, we dont need to cache any gradients
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "episode" in infos:
                for idx, return_val in enumerate(infos["episode"]["r"]):
                    if return_val == 0:
                        continue
                    # print(f"global_step={global_step}, episodic_return={return_val}")
                    writer.add_scalar(
                        f"charts/episodic_return/env_{idx}",
                        return_val,
                        global_step,
                    )
                for idx, return_val in enumerate(infos["episode"]["l"]):
                    if return_val == 0:
                        continue
                    writer.add_scalar(
                        f"charts/episodic_return/env_{idx}",
                        return_val,
                        global_step,
                    )

        # GAE
        # ppo bootstrapes values if the envs are not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
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
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )  #
                returns = advantages + values  # advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = (
                        rewards[t] + args.gamma * nextnonterminal * next_return
                    )  # sum of discounted rewards, see difference to gae
                advantages = returns - values

        # flatten the batch, stores flattened storage values
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # to do training, we require all the indices of the batch
        # Optimizing the policy and the value networks
        b_inds = np.arange(args.batch_size)

        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # shuffle these batch indices
            for start in range(0, args.batch_size, args.minibatch_size):
                # loop through entire batch one minibatch at a time
                # each minibatch indices contains 128 items of their randomized batch indices
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # now training begins
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )  # forward pass on a minibatch observation
                # b_actions.long()[mb_inds] (pass in minibatched actions, so that the agent does not sample any new actions)
                logratio = (
                    newlogprob - b_logprobs[mb_inds]
                )  # newlogprobs - old_logprobs
                ratio = (
                    logratio.exp()
                )  # during the first forward pass, this ratio would only contain ones, because we havent made changes to logprobabilities

                # Debug variables
                with torch.no_grad():
                    # calculate approx_kl
                    old_approx_kl = (
                        -logratio
                    ).mean()  # helps to understand how aggresively the policy upgrades
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]

                mb_advantages = b_advantages[mb_inds]  # minibatch advantages
                if (
                    args.norm_adv
                ):  # normalization, with small scalar value to prevent divide by zero error
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                # max of negatives (equivalent Paper min of positives L CLIP)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:  # original implementation
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:  # normally
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = (
                    entropy.mean()
                )  # measure of chaos in a action probability distribution
                # maximizing entropy -> agent explores more
                # minimize policy loss and value loss, maximize entropy loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )  # this implemetation detail (11) adds this to backpropagation
                optimizer.step()

            # break update epochs
            # also possible to implement this at the minibatch level (here batch level)
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Debug variable: explain variance, tells you wheter value function is a good estimator of return
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
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
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

    # save the trained model
    torch.save(agent.state_dict(), f"models/{run_name}_agent.pt")
    print(f"Model saved to models/{run_name}_agent.pt")

    envs.close()  # underlying error remains
    # AttributeError: 'RecordVideo' object has no attribute 'enabled'

    writer.close()
