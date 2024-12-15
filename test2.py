def collect_rollout_data(self):
    # Collect rollout data at each step
    # One trajectorie per env

    for step in range(0, self.args.num_steps):
        self.global_step += self.args.num_envs
        self.obs[step] = self.next_obs
        self.dones[step] = self.next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
            self.values[step] = value.flatten()
        self.actions[step] = action
        self.logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, self.true_rewards, terminations, truncations, infos = self.envs.step(
            action.cpu().numpy()
        )

        state_action_pairs = np.hstack([self.next_obs, action.cpu().numpy()])
        # with torch.no_grad():
        self.predicted_rewards = self.reward_model(torch.tensor(state_action_pairs))

        if test:
            if step == 0:
                state_action_pairs = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
                self.true_rewards = np.array([0.01, 0.03])
                self.predicted_rewards = torch.tensor(np.array([[0.05], [0.07]]))

            if step == 1:
                state_action_pairs = np.array([[6, 7, 8, 9, 10], [16, 17, 18, 19, 20]])
                self.true_rewards = np.array([0.02, 0.04])
                self.predicted_rewards = torch.tensor(np.array([[0.06], [0.08]]))
            if step == 2:
                state_action_pairs = np.array(
                    [[100, 101, 102, 103, 104], [105, 106, 107, 108, 109]]
                )
                self.true_rewards = np.array([7.3, 1.1])
                self.predicted_rewards = torch.tensor(np.array([[6.25], [2.75]]))

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

        if test:
            if step == 2:
                input_dim = 5
                self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(
                    self.args.num_envs, -1, input_dim
                )
                self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(
                    -1, input_dim
                )
                self.true_reward_buffer = self.true_reward_buffer.reshape(-1)
                self.predicted_rewards_buffer = self.predicted_rewards_buffer.reshape(
                    -1
                )
                print(self.obs_action_pair_buffer)
                print(self.true_reward_buffer)
                print(self.predicted_rewards_buffer)
                return

    obs_dim = np.prod(self.envs.single_observation_space.shape)
    action_dim = np.prod(self.envs.single_action_space.shape)
    input_dim = obs_dim + action_dim
    self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(
        self.args.num_envs, -1, input_dim
    )

    self.obs_action_pair_buffer = self.obs_action_pair_buffer.reshape(-1, input_dim)
    self.true_reward_buffer = self.true_reward_buffer.reshape(-1)
    self.predicted_rewards_buffer = self.predicted_rewards_buffer.reshape(-1)

    # # print(self.obs_action_pair_buffer)
    # print(self.obs_action_pair_buffer.shape)
    # print(self.true_reward_buffer.shape)
    # # print(self.true_reward_buffer)
    # print(self.predicted_rewards_buffer.shape)
    # raise Exception
