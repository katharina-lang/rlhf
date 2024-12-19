import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from rlhf.configs.arguments import Args
from rlhf.core.agent import Agent
from rlhf.utils.env import make_env
from rlhf.core.reward_model import RewardModel
from rlhf.core.labeling import Labeling
from rlhf.core.ppo_setup import PPOSetup


class PPO:
    def __init__(self, run_name, args, test_data=None):
        self.args = args
        # Rollouts Data
        args.batch_size = int(self.args.num_envs * self.args.num_steps)
        args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)
        # Number of policy updates
        args.num_iterations = self.args.total_timesteps // self.args.batch_size

        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        self.writer = PPOSetup.set_up_writer(run_name, self.args)

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        self.device = PPOSetup.set_up_device(self.args)

        self.envs = PPOSetup.set_up_envs(self.args, run_name)

        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=1e-5
        )

        self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values = (
            PPOSetup.set_up_storage(args, self.envs, self.device)
        )

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs, _ = self.envs.reset(seed=args.seed)
        self.next_obs = torch.Tensor(self.next_obs).to(self.device)
        self.next_done = torch.zeros(args.num_envs).to(self.device)

        # reward model setup
        self.segment_size = self.args.segment_size
        obs_dim = np.prod(self.envs.single_observation_space.shape)
        action_dim = np.prod(self.envs.single_action_space.shape)
        input_dim = obs_dim + action_dim

        self.reward_model = RewardModel(input_dim=input_dim).to(self.device)
        self.reward_optimizer = optim.Adam(
            self.reward_model.parameters(), lr=1e-3, weight_decay=1e-5
        )

        self.labeled_data = []
        self.obs_action_pair_buffer = None
        self.env_reward_buffer = None
        self.predicted_rewards_buffer = None

        # Falls Testdaten vorhanden
        if test_data:
            self.obs_action_pair_buffer, self.true_reward_buffer, self.predicted_rewards_buffer = test_data


    def collect_rollout_data(self):

        for step in range(0, self.args.num_steps):
            self.global_step += self.args.num_envs
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    self.next_obs
                )
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, self.env_reward, terminations, truncations, infos = (
                self.envs.step(action.cpu().numpy())
            )
            print("Next obs" + str(next_obs))

            state_action_pairs = np.hstack([self.next_obs, action.cpu().numpy()])
            with torch.no_grad():
                self.predicted_reward = self.reward_model(
                    torch.tensor(state_action_pairs)
                )

            self.save_data(state_action_pairs)

            # Data Storage (cleanrl)
            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = (
                torch.tensor(self.predicted_reward).to(self.device).view(-1)
            )
            self.next_obs, self.next_done = torch.Tensor(next_obs).to(
                self.device
            ), torch.Tensor(self.next_done).to(self.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={self.global_step}, episodic_return={info['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            self.global_step,
                        )

        self.reshape_data()

    def save_data(self, state_action_pairs):
        if self.obs_action_pair_buffer is None:
           self.obs_action_pair_buffer = state_action_pairs
        else:
            self.obs_action_pair_buffer = np.hstack(
                [self.obs_action_pair_buffer, state_action_pairs]
            )

        self.env_reward = self.env_reward.reshape(self.args.num_envs, 1)

        if self.env_reward_buffer is None:
            self.env_reward_buffer = self.env_reward
        else:
            self.env_reward_buffer = np.hstack(
                [self.env_reward_buffer, self.env_reward]
            )

        if self.predicted_rewards_buffer is None:
            self.predicted_rewards_buffer = self.predicted_reward

        else:
            self.predicted_rewards_buffer = torch.cat(
                [self.predicted_rewards_buffer, self.predicted_reward], dim=1
            )

    def reshape_data(self):
        obs_dim = np.prod(self.envs.single_observation_space.shape)
        action_dim = np.prod(self.envs.single_action_space.shape)
        input_dim = obs_dim + action_dim
        self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(
            self.args.num_envs, -1, input_dim
        )
        self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(-1, input_dim)
        self.env_reward_buffer = self.env_reward_buffer.reshape(-1)
        self.predicted_rewards_buffer = self.predicted_rewards_buffer.reshape(-1)

    def advantage_calculation(
        self,
    ):
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = (
                    self.rewards[t]
                    + self.args.gamma * nextvalues * nextnonterminal
                    - self.values[t]
                )
                self.advantages[t] = lastgaelam = (
                    delta
                    + self.args.gamma
                    * self.args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            self.returns = self.advantages + self.values


    def record_rewards_for_plotting_purposes(self, explained_var):
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar(
            "charts/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            self.global_step,
        )
        self.writer.add_scalar(
            "losses/value_loss", self.agent.v_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/policy_loss", self.agent.pg_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/entropy", self.agent.entropy_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/old_approx_kl", self.agent.old_approx_kl.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/approx_kl", self.agent.approx_kl.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/clipfrac", np.mean(self.agent.clipfracs), self.global_step
        )
        self.writer.add_scalar(
            "losses/explained_variance", explained_var, self.global_step
        )
        print("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar(
            "charts/SPS",
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step,
        )

        