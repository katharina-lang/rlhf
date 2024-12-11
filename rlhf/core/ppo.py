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


class PPO:
    def __init__(self, run_name, args, reward_model=None):
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
        if reward_model:
            self.segment_size = 60
            obs_dim = np.prod(self.envs.single_observation_space.shape)
            action_dim = np.prod(self.envs.single_action_space.shape)
            input_dim = obs_dim + action_dim

            self.reward_model = RewardModel(input_dim=input_dim).to(self.device)
            self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
            # Store preference data (pairs of trajectories)
            self.trajectory_database = (
                []
            )  # should this be reset after every reward model training?
            self.trajectory_buffers = [[] for _ in range(self.args.num_envs)]
            self.labeled_data = []

            self.obs_action_pair_buffer = None
            self.true_reward_buffer = None
            self.predicted_rewards_buffer = None

    def collect_rollout_data(self):
        obs_action_pair_list = []
        true_reward_list = []
        predicted_reward_list = []
        # Collect rollout data at each step
        # One trajectorie per env
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
            next_obs, self.true_rewards, terminations, truncations, infos = (
                self.envs.step(action.cpu().numpy())
            )

            state_action_pairs = np.hstack([self.next_obs, action.cpu().numpy()])
            with torch.no_grad():
                self.predicted_rewards = self.reward_model(
                    torch.tensor(state_action_pairs)
                )

            if self.obs_action_pair_buffer is None:
                self.obs_action_pair_buffer = state_action_pairs
            else:
                self.obs_action_pair_buffer = np.hstack(
                    [self.obs_action_pair_buffer, state_action_pairs]
                )

            self.true_rewards = self.true_rewards.reshape(self.args.num_envs, 1)

            if self.true_reward_buffer is None:
                self.true_reward_buffer = self.true_rewards
            else:
                self.true_reward_buffer = np.hstack(
                    [self.true_reward_buffer, self.true_rewards]
                )

            if self.predicted_rewards_buffer is None:
                self.predicted_rewards_buffer = self.predicted_rewards.numpy()
            else:

                self.predicted_rewards_buffer = np.hstack(
                    [self.predicted_rewards_buffer, self.predicted_rewards.numpy()]
                )

            # Data Storage
            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = (
                torch.tensor(self.predicted_rewards).to(self.device).view(-1)
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

        obs_dim = np.prod(self.envs.single_observation_space.shape)
        action_dim = np.prod(self.envs.single_action_space.shape)
        input_dim = obs_dim + action_dim
        self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(
            self.args.num_envs, -1, input_dim
        )
        # habe 2 envs, und für jeden der 2048 steps steht eine liste mit den obs_actions
        # flach machen, so das es nur noch eine trajektorie gibt
        self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(1, -1, 23)
        self.true_reward_buffer = self.true_reward_buffer.reshape(1, -1)
        self.predicted_rewards_buffer = self.predicted_rewards_buffer.reshape(1, -1)

        # print(self.obs_action_pair_buffer)
        # print(self.obs_action_pair_buffer.shape)
        # print(self.true_reward_buffer.shape)
        # print(self.predicted_rewards_buffer.shape)

        # raise Exception

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

    def optimize_agent_and_critic(self):

        self.b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        self.b_logprobs = self.logprobs.reshape(-1)
        self.b_actions = self.actions.reshape(
            (-1,) + self.envs.single_action_space.shape
        )
        self.b_advantages = self.advantages.reshape(-1)
        self.b_returns = self.returns.reshape(-1)
        self.b_values = self.values.reshape(-1)

        self.b_inds = np.arange(self.args.batch_size)
        self.clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(self.b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = self.b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    self.b_obs[mb_inds], self.b_actions[mb_inds]
                )
                logratio = newlogprob - self.b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    self.old_approx_kl = (-logratio).mean()
                    self.approx_kl = ((ratio - 1) - logratio).mean()
                    self.clipfracs += [
                        ((ratio - 1.0).abs() > self.args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                self.mb_advantages = self.b_advantages[mb_inds]
                if self.args.norm_adv:
                    self.mb_advantages = (
                        self.mb_advantages - self.mb_advantages.mean()
                    ) / (self.mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -self.mb_advantages * ratio
                pg_loss2 = -self.mb_advantages * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                )
                self.pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - self.b_returns[mb_inds]) ** 2
                    v_clipped = self.b_values[mb_inds] + torch.clamp(
                        newvalue - self.b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    self.v_loss = 0.5 * v_loss_max.mean()
                else:
                    self.v_loss = (
                        0.5 * ((newvalue - self.b_returns[mb_inds]) ** 2).mean()
                    )

                self.entropy_loss = entropy.mean()
                loss = (
                    self.pg_loss
                    - self.args.ent_coef * self.entropy_loss
                    + self.v_loss * self.args.vf_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()

            if self.args.target_kl is not None and self.approx_kl > self.args.target_kl:
                break

    def record_rewards_for_plotting_purposes(self, explained_var):
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar(
            "charts/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            self.global_step,
        )
        self.writer.add_scalar(
            "losses/value_loss", self.v_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/policy_loss", self.pg_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/entropy", self.entropy_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/old_approx_kl", self.old_approx_kl.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/approx_kl", self.approx_kl.item(), self.global_step
        )
        self.writer.add_scalar(
            "losses/clipfrac", np.mean(self.clipfracs), self.global_step
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

    def train_reward_model(self):
        return
        """Train reward model from preferences."""
        if not self.labeled_data:
            print("No labeled data available for training.")
            return

        self.reward_model.train()
        optimizer = self.reward_optimizer

        # Anzahl der Trainingsepochen
        epochs = 10
        batch_size = 32

        for epoch in range(epochs):
            random.shuffle(self.labeled_data)
            for i in range(0, len(self.labeled_data), batch_size):
                batch = self.labeled_data[i : i + batch_size]
                inputs1, inputs2, preferences = zip(*batch)
                preferences = torch.tensor(preferences, dtype=torch.float32).to(
                    self.device
                )

                # Segmente vorbereiten
                rewards1 = [
                    torch.sum(
                        self.reward_model(
                            torch.tensor(segment, dtype=torch.float32)
                            .view(1, -1)
                            .to(self.device)
                        )
                    )
                    for segment in inputs1
                ]
                rewards2 = [
                    torch.sum(
                        self.reward_model(
                            torch.tensor(segment, dtype=torch.float32)
                            .view(1, -1)
                            .to(self.device)
                        )
                    )
                    for segment in inputs2
                ]

                rewards1 = torch.stack(rewards1)  # [batch_size]
                rewards2 = torch.stack(rewards2)  # [batch_size]

                # Wahrscheinlichkeiten gemäß Bradley-Terry-Modell
                probabilities = torch.sigmoid(rewards1 - rewards2)

                # Kreuzentropie-Verlust
                loss = -torch.mean(
                    preferences[:, 0] * torch.log(probabilities + 1e-8)
                    + preferences[:, 1] * torch.log(1 - probabilities + 1e-8)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Nach dem Training die gelabelten Daten zurücksetzen
            self.labeled_data = []

    def preference_elicitation(self, segment_one, segment_two):
        segment_obs_actionOne, true_rewardOne, predicted_rewardOne = segment_one
        segment_obs_actionTwo, true_rewardTwo, predicted_rewardTwo = segment_two

        if true_rewardOne > true_rewardTwo:
            labelOne = 1
            labelTwo = 0
        elif true_rewardTwo > true_rewardOne:
            labelOne = 0
            labelTwo = 1
        else:
            labelOne = labelTwo = 0.5

        return (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (predicted_rewardOne, predicted_rewardTwo),
        )

    def select_segments(self):

        self.obs_action_pair_buffer
        self.true_reward_buffer
        self.predicted_rewards_buffer
        data_points = self.true_reward_buffer.shape[1]

        # segment amount
        segment_amount = data_points // self.segment_size

        segments = []
        for _ in range(segment_amount):
            start_idx = np.random.randint(0, data_points - self.segment_size)
            end_idx = start_idx + self.segment_size
            segment_obs_action = self.obs_action_pair_buffer[start_idx:end_idx]
            true_reward = sum(self.true_reward_buffer[start_idx:end_idx])
            predicted_reward = sum(self.predicted_rewards_buffer[start_idx:end_idx])
            segment = (segment_obs_action, true_reward, predicted_reward)
            segments.append(segment)

        self.obs_action_pair_buffer = None
        self.true_reward_buffer = None
        self.predicted_rewards_buffer = None

        return segments

        self.obs_action_pair_buffer = self.obs_action_pair_buffer.flatten()
        self.true_reward_buffer = self.true_reward_buffer.flatten()
        self.predicted_rewards_buffer = self.predicted_rewards_buffer.flatten()

        db_length = len(self.trajectory_database)
        segments = []

        for id in range(db_length):
            start_idx = np.random.randint(
                0, len(self.trajectory_database[id]) - self.segment_size
            )
            if len(self.trajectory_database[id]) < self.segment_size:
                continue

            segment = self.trajectory_database[id][
                start_idx : start_idx + self.segment_size
            ]
            segment_rewards = sum([snapshot["reward"] for snapshot in segment])
            # for snapshot in segment:
            #     print(snapshot["action"])
            segment_obs_action_matrix = [
                np.concatenate([snapshot["obs"], snapshot["action"]])
                for snapshot in segment
            ]

            segment_flat = np.array(segment_obs_action_matrix).flatten()
            segments.append((segment_flat, segment_rewards))

            # segments.append((segment_obs_action_matrix, segment_rewards))
            # vielleicht jeweils noch die summierten predicteten rewards zurückgeben # TODO
        return segments

    def get_labeled_data(self):
        """
        one element of labeld_data looks like the following:
        (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (predicted_rewardOne, predicted_rewardTwo),
        )
        where the obs_action is the input for the reward model
        and predicted_rewardOne is the total reward for segment One
        """
        segments = self.select_segments()
        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            segments_label_reward = self.preference_elicitation(
                segment_one, segment_two
            )  # kann hier auch einfach das triple zurückgeben
            self.labeled_data.append(segments_label_reward)

        # nach jeden labeln reseten
        self.trajectory_database = []

        # label nach jedem training reseten

        # should this be reset after every training?

        # print(segment_one)
        # # i expect this to be a tuple , (input, rewrds)
        # break


class PPOSetup:
    @staticmethod
    def set_up_writer(run_name, args):
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        return writer

    @staticmethod
    def set_up_device(args):
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        device = "cpu"
        return device

    @staticmethod
    def set_up_envs(args, run_name):
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
                for i in range(args.num_envs)
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        return envs

    def set_up_storage(args, envs, device):
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
        return obs, actions, logprobs, rewards, dones, values
