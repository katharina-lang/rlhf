import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rlhf.core.agent import Agent
from rlhf.core.reward_model import RewardModel
from rlhf.core.ppo_setup import PPOSetup
from scipy.stats import pearsonr
import rlhf.core.unsupervised_pt as up
from rlhf.core.unsupervised_pt import compute_intrinsic_reward


class PPO:
    def __init__(self, run_name, args):
        """
        Initializes the PPO training setup.

        Args:
            run_name (str): Identifier for the training run.
            args (Namespace): Configuration arguments for the PPO setup.
        """
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
        obs_dim = np.prod(self.envs.single_observation_space.shape)
        action_dim = np.prod(self.envs.single_action_space.shape)
        input_dim = obs_dim + action_dim

        self.reward_models = [
            RewardModel(input_dim=input_dim, dropout_p=args.dropout).to(self.device)
            for _ in range(args.num_models)
        ]

        self.optimizers = [
            optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            for model in self.reward_models
        ]

        if self.args.unsupervised_pretraining:
            self.density_model = up.KNNDensityModel(k=5)

    def collect_rollout_data(self, unsupervised_pretraining=False):
        """
        Collects data from the environment by running the policy.

        Args:
            unsupervised_pretraining (bool, optional): If true, intrinsic motivation is used instead of external rewards.
        """
        self.obs_action_pair_buffer = None
        self.env_reward_buffer = None
        self.predicted_rewards_buffer = None

        if unsupervised_pretraining:
            iteration_reward = 0

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

            if unsupervised_pretraining:
                self.density_model.add_states(
                    self.next_obs.cpu().numpy()
                )  # Add observed states
                self.env_reward = torch.tensor(
                    [
                        compute_intrinsic_reward(state, self.density_model)
                        for state in next_obs
                    ],
                    dtype=torch.float32,
                ).to(self.device)
                iteration_reward += self.env_reward

            state_action_pairs = np.hstack(
                [self.next_obs.cpu().numpy(), action.cpu().numpy()]
            )
            state_action_tensor = torch.tensor(state_action_pairs, device=self.device)

            if not unsupervised_pretraining:
                with torch.no_grad():
                    predictions = []
                    for model in self.reward_models:
                        pred = model(state_action_tensor)
                        predictions.append(pred)
                    self.predicted_reward = torch.mean(torch.stack(predictions), dim=0)

            self.save_data(state_action_pairs, unsupervised_pretraining)

            # Data Storage (cleanrl)
            self.next_done = np.logical_or(terminations, truncations)
            if not unsupervised_pretraining:
                self.rewards[step] = (
                    torch.tensor(self.predicted_reward).to(self.device).view(-1)
                )
            else:
                self.rewards[step] = (
                    self.env_reward.clone().detach().to(self.device).view(-1)
                )
            self.next_obs, self.next_done = torch.Tensor(next_obs).to(
                self.device
            ), torch.Tensor(self.next_done).to(self.device)

            # new gym api
            if "episode" in infos:
                eps = list(infos["episode"]["_t"])
                done_envs = [
                    i for i, finished in enumerate(eps) if finished is np.True_
                ]
                for i in done_envs:
                    print(
                        f"global_step={self.global_step}, episodic_return={infos['episode']['r'][i]}"
                    )

                    self.writer.add_scalar(
                        "metrics/episodic_return",
                        infos["episode"]["r"][i],
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "metrics/episodic_length",
                        infos["episode"]["l"][i],
                        self.global_step,
                    )

            # old gym api
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={self.global_step}, episodic_return={info['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "metrics/episodic_return",
                            info["episode"]["r"],
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "metrics/episodic_length",
                            info["episode"]["l"],
                            self.global_step,
                        )

        if unsupervised_pretraining:
            print("Iteration_reward: " + str(iteration_reward))

        self.reshape_data(unsupervised_pretraining)

        if not unsupervised_pretraining:
            self.track_pearsonr(self.env_reward_buffer, self.predicted_rewards_buffer)

    def save_data(self, state_action_pairs, unsupervised_pretraining):
        """
        Stores state-action pairs and rewards into their respective buffers.

        Args:
            state_action_pairs (np.array): The collected observation-action pairs.
            unsupervised_pretraining (bool): Indicates whether unsupervised pretraining is used.
        """
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

        if not unsupervised_pretraining:
            if self.predicted_rewards_buffer is None:
                self.predicted_rewards_buffer = self.predicted_reward

            else:
                self.predicted_rewards_buffer = torch.cat(
                    [self.predicted_rewards_buffer, self.predicted_reward], dim=1
                )

    def reshape_data(self, unsupervised_pretraining):
        """
        Reshapes data buffers to match the expected input dimensions.

        Args:
            unsupervised_pretraining (bool): Indicates whether unsupervised pretraining is used.
        """
        obs_dim = np.prod(self.envs.single_observation_space.shape)
        action_dim = np.prod(self.envs.single_action_space.shape)
        input_dim = obs_dim + action_dim
        self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(-1, input_dim)
        self.env_reward_buffer = self.env_reward_buffer.reshape(-1)
        if not unsupervised_pretraining:
            self.predicted_rewards_buffer = self.predicted_rewards_buffer.reshape(-1)

    def advantage_calculation(
        self,
    ):
        """
        Computes the Generalized Advantage Estimation (GAE) for policy optimization.
        """
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
        """
        Records training metrics for visualization and logs them using TensorBoard.

        Args:
            explained_var (float): Explained variance of value function predictions.
        """
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

    def track_pearsonr(self, env_rewards, predicted_rewards):
        """
        Computes and logs the Pearson correlation coefficient between environment rewards and predicted rewards.

        Args:
            env_rewards (array-like): True rewards from the environment.
            predicted_rewards (array-like): Model-predicted rewards.
        """
        if torch.is_tensor(predicted_rewards):
            predicted_rewards = predicted_rewards.cpu().numpy()

        pearson_corr, _ = pearsonr(env_rewards, predicted_rewards)

        self.writer.add_scalar(
            "metrics/pearson_correlation", pearson_corr, self.global_step
        )
