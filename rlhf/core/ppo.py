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
        args.batch_size = int(self.args.num_envs * self.args.num_steps)
        args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)
        args.num_iterations = self.args.total_timesteps // self.args.batch_size

        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        self.writer = PPOSetup.set_up_writer(run_name, self.args)

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

        self.global_step = 0
        self.start_time = time.time()
        self.next_obs, _ = self.envs.reset(seed=args.seed)
        self.next_obs = torch.Tensor(self.next_obs).to(self.device)
        self.next_done = torch.zeros(args.num_envs).to(self.device)

        # reward model setup
        if reward_model:
            self.segment_size = 2       # segment_size für Trainingszwecke von 60 auf 2 gesetzt
            obs_dim = np.prod(self.envs.single_observation_space.shape)
            action_dim = np.prod(self.envs.single_action_space.shape)
            input_dim = obs_dim + action_dim

            self.reward_model = RewardModel(input_dim=input_dim).to(self.device)
            self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
            self.trajectory_database = []
            self.trajectory_buffers = [[] for _ in range(self.args.num_envs)]
            self.labeled_data = []

            self.obs_action_pair_buffer = None
            self.true_reward_buffer = None
            self.predicted_rewards_buffer = None

    def collect_rollout_data(self):
        for step in range(self.args.num_steps):
            self.global_step += self.args.num_envs
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                self.values[step] = value.flatten()

            self.actions[step] = action
            self.logprobs[step] = logprob

            next_obs, self.true_rewards, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            state_action_pairs = np.hstack([self.next_obs.cpu().numpy(), action.cpu().numpy()])

            # Store predicted rewards using the current reward model (fixed for this rollout)
            self.predicted_rewards = self.compute_predicted_rewards(state_action_pairs)
            self.update_buffers(state_action_pairs, self.true_rewards, self.predicted_rewards)

            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(self.predicted_rewards).to(self.device).view(-1)
            self.next_obs, self.next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(self.next_done).to(self.device)

            if "final_info" in infos:
                self.log_episode_info(infos)

    def compute_predicted_rewards(self, state_action_pairs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return (
                self.reward_model(torch.tensor(state_action_pairs, dtype=torch.float32).to(self.device))
                .cpu()
                .numpy()
            )

    def update_buffers(self, state_action_pairs, true_rewards, predicted_rewards):
        self.obs_action_pair_buffer = (
            np.vstack([self.obs_action_pair_buffer, state_action_pairs])
            if self.obs_action_pair_buffer is not None
            else state_action_pairs
        )
        self.true_reward_buffer = (
            np.vstack([self.true_reward_buffer, true_rewards])
            if self.true_reward_buffer is not None
            else true_rewards
        )
        self.predicted_rewards_buffer = (
            np.vstack([self.predicted_rewards_buffer, predicted_rewards])
            if self.predicted_rewards_buffer is not None
            else predicted_rewards
        )

    def train_reward_model(self):
        if not self.labeled_data:
            print("Keine gelabelten Daten vorhanden. Training übersprungen.")
            return

        self.reward_model.train()
        optimizer = self.reward_optimizer

        epochs = 10
        batch_size = 32

        for epoch in range(epochs):
            random.shuffle(self.labeled_data)
            for i in range(0, len(self.labeled_data), batch_size):
                batch = self.labeled_data[i:i + batch_size]
                inputs1, inputs2, labels = zip(*batch)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

                rewards1 = [
                    torch.sum(
                        torch.tensor(segment[2], dtype=torch.float32).to(self.device)
                    )
                    for segment in inputs1
                ]
                rewards2 = [
                    torch.sum(
                        torch.tensor(segment[2], dtype=torch.float32).to(self.device)
                    )
                    for segment in inputs2
                ]

                rewards1 = torch.stack(rewards1)
                rewards2 = torch.stack(rewards2)

                probabilities = torch.sigmoid(rewards1 - rewards2)

                loss = -torch.mean(
                    labels * torch.log(probabilities + 1e-8)
                    + (1 - labels) * torch.log(1 - probabilities + 1e-8)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.labeled_data = []

    def select_segments(self):
        segments_per_env = [[] for _ in range(self.args.num_envs)]

        for env_id in range(self.args.num_envs):
            steps = self.obs_action_pair_buffer[env_id]
            true_rewards = self.true_reward_buffer[env_id]
            predicted_rewards = self.predicted_rewards_buffer[env_id]

            num_steps = steps.shape[0]

            """
            if num_steps < self.segment_size:
                print(f"Env {env_id}: Not enough steps to create segments.")
                continue
            """

            num_segments = num_steps // self.segment_size

            for _ in range(num_segments):
                start_idx = np.random.randint(0, num_steps - self.segment_size)
                end_idx = start_idx + self.segment_size

                segment_obs_action = steps[start_idx:end_idx]
                segment_true_reward = np.sum(true_rewards[start_idx:end_idx])
                segment_predicted_reward = np.sum(predicted_rewards[start_idx:end_idx])

                segment = (segment_obs_action, segment_true_reward, segment_predicted_reward)
                segments_per_env[env_id].append(segment)
        return segments_per_env


    def label_segments(self):
        labeled_triplets = []
        segments_per_env = self.select_segments()

        for env_id, segments in enumerate(segments_per_env):
            if len(segments) < 2:
                continue

            for _ in range(len(segments) // 2):
                segment_1 = segments.pop(random.randint(0, len(segments) - 1))
                segment_2 = segments.pop(random.randint(0, len(segments) - 1))

                true_reward_1 = segment_1[1]
                true_reward_2 = segment_2[1]

                if true_reward_1 > true_reward_2:
                    label = 0
                elif true_reward_1 < true_reward_2:
                    label = 1
                else:
                    label = 0.5

                triplet = (segment_1, segment_2, label)
                labeled_triplets.append(triplet)

        self.labeled_data.extend(labeled_triplets)


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
