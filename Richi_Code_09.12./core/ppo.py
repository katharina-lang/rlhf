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
    def __init__(self, run_name, args, reward_model=None):          #reward-model ist kein Boolean, sondern tatsächlich ein Reward Model: truthy, falls eins übergeben wird, falsy falls nicht
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

        # reward model setup                                                    #Falls wir angegeben haben, dass wir mit eigenem Reward-Model (ohne Umgebungs-Rewards) trainieren wollen: Initialisierung
        if reward_model:                                                        #des neuronalen Netzes und des Speichers der Trajectories und Präferenzen --> wird in Arrays gespeichert
            self.reward_model = RewardModel(
                input_dim=np.prod(self.envs.single_observation_space.shape)     # Frage: Hier haben wir ja den Observation Space Shape als Größe der Eingabe. Müssen wir nicht Obs und Action zusammen eingeben?
            ).to(self.device)
            self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
            # Store preference data (pairs of trajectories)
            self.trajectory_database = (
                []
            )  # should this be reset after every reward model training?     # Nein, oder? Wir wollen aus alten Erfahrungen ja weiterhin lernen.
            self.trajectory_buffers = [[] for _ in range(self.args.num_envs)]
            self.segment_size = 60
            self.labeled_data = []

    def collect_rollout_data(self):
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
            next_obs, self.env_rewards, terminations, truncations, infos = (   # in der Liste (ein Index entspricht einer Umgebung) env_rewards wird je Umgebung der Reward des Schrittes gespeichert
                self.envs.step(action.cpu().numpy())
            )

            # Append data to trajectory buffers                 # Für jedes Environment hängen wir die Daten des aktuellen Schrittes an unsere Liste an aufeinanderfolgenden Schritten an
            for env_idx in range(self.args.num_envs):
                self.trajectory_buffers[env_idx].append(
                    {
                        "obs": self.next_obs[env_idx].cpu().numpy(),        # Daten des aktuellen Schrittes werden als Dictionary gespeichert
                        "action": action[env_idx].cpu().numpy(),            # TO DO: predicted reward mit speichern
                        "reward": self.env_rewards[env_idx],
                    }
                )

                # If the environment resets, save the trajectory and start a new one        # Wenn in irgendeiner Umgebung die Episode endet, wird der vollendete Trajectory an die Liste der Trajectories
                                                                                            # angehängt und der jeweilige "Schrittesammler" dieses Trajectories geleert
                if terminations[env_idx] or truncations[env_idx]:
                    trajectories = self.trajectory_buffers[env_idx]
                    self.trajectory_database.append(trajectories)
                    self.trajectory_buffers[env_idx] = []

                # If reward model is provided
                if self.reward_model:           # prüft, ob ein reward model übergeben wurde --> falls ein gültiges Objekt übergeben wurde, ist self.reward_model ein "truthy-Wert"
                    with torch.no_grad():
                        predicet_reward = self.reward_model(            # forward(torch.Tensor((next_obs[env_idx],action[env_idx]).to(self.device) ??
                            torch.Tensor(next_obs).to(self.device)
                        )
                    reward = predicet_reward.cpu().numpy().squeeze()    # reward ist Liste (ein Index entspricht einer Umgebung): Reward des Schrittes ist je Umgebung gespeichert
                                                                        # --> sind entweder die von der Umgebung zurückgegebenen Rewards oder die vom Reward-Model geschätzten Rewards
                else:
                    reward = self.env_rewards

            # Data Storage
            self.next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)      # Rewards werden an die Stelle step in die Reward-Matrix gespeichert --> ab jetzt kann unsere Policy ganz normal
                                                                                    # geupdated werden, da das Update ja genau von dieser Matrix abhängt (und auch nur von dieser)
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

    def advantage_calculation(
        self,
    ):
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:            # War es der letzte Schritt in unserem Batch?
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value                 # --> Dann haben wir ja keinen Value des nächsten States, den wir auf den Reward addieren können, weil es gar keinen nächsten State gibt
                                                            # Wir wollen das Loss aber trzd berechnen können --> wir nehmen als Value des nächsten States einfach den Value dieses States
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
        """Train reward model from preferences."""

        self.trajectory_database = []
        self.labeled_data = []

    def preference_elicitation(self, segment_one, segment_two):
        inputOne, rewardOne = segment_one           # Segment_one ist ja gerade ein Paar aus (State/Aktion) und Reward --> können wir einzeln zuweisen
        inputTwo, rewardTwo = segment_two

        if rewardOne > rewardTwo:
            labelOne = 1
            labelTwo = 0
        elif rewardTwo > rewardOne:
            labelOne = 0
            labelTwo = 1
        else:
            labelOne = labelTwo = 0.5

        return (inputOne, labelOne), (inputTwo, labelTwo)

"""     if rewardOne > rewardTwo:
            label = 0
        elif rewardTwo > rewardOne:
            label = 1
        else:
            label = 0.5

        return (inputOne, inputTwo, Label)
"""

    def select_segments(self):
        db_length = len(self.trajectory_database)       # Wie viele Trajectories haben wir insgesamt?
        segments = []

        for id in range(db_length):
            start_idx = np.random.randint(
                0, len(self.trajectory_database[id]) - self.segment_size    # kann nicht an einer späteren Stelle starten als Länge d. Trajectories - Größe Segement, da sonst Segment nicht "reinpasst"
            )
            if len(self.trajectory_database[id]) < self.segment_size:       # falls der ganze Trajectory kleiner als ein Segment ist, soll zum nächsten Trajectory gesprungen werden
                continue

            segment = self.trajectory_database[id][
                start_idx : start_idx + self.segment_size
            ]
            segment_rewards = sum([snapshot["reward"] for snapshot in segment])     # alle Rewards des ausgewählten Segments aufsummieren --> kumulativer Reward
            # for snapshot in segment:
            #     print(snapshot["action"])
            segment_obs_action_matrix = [
                np.concatenate([snapshot["obs"], snapshot["action"]])               # für jeden Schritt in unserem Segment konkatenieren wir die jeweilige Beobachtung mit der daraufhin genommenen Aktion
                for snapshot in segment                                             # --> wir erhalten eine Matrix mit den Schritten als Zeilen und den 2 Spalten obs und action
            ]
            segments.append((segment_obs_action_matrix, segment_rewards))

        return segments             # select segments gibt uns eine Liste an Segmenten (bestehend aus obs/action-Paaren) und deren kumulative Rewards zurück
                                    # ein Listen-Element ist immer ein Paar aus Segment und kumulativem Reward dieses Segments

                                    # Frage: Wir haben dann eine Liste, die vorne jeweils ein Segment der vorderen Trajektorien hat und hinten die der hinteren. Verhindert pop() dann nicht die Durchmischung?
                                    # Wir vergleichen ja dann immer Segmente aus aufeinanderfolgenden Trajektorien...

    def get_labeled_data(self):         # Wählt immer zwei Segmente aus unserer Liste an Segmenten aus, bis diese leer ist. Dann werden diese Paare gelabelt und an unsere Liste an gelabelten Daten angehängt
        segments = self.select_segments()
        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            labeledOne, labeledTwo = self.preference_elicitation(
                segment_one, segment_two
            )
            self.labeled_data.append(labeledOne)
            self.labeled_data.append(labeledTwo)
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
