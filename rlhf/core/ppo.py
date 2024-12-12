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

    def log_episode_info(self, infos):
        for info in infos["final_info"]:
            if info and "episode" in info:
                episodic_return = info["episode"]["r"]
                episodic_length = info["episode"]["l"]

                print(f"global_step={self.global_step}, episodic_return={episodic_return}")

                self.writer.add_scalar("charts/episodic_return", episodic_return, self.global_step)
                self.writer.add_scalar("charts/episodic_length", episodic_length, self.global_step)



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

    # Zugriff auf die gelabelten Daten: Tripel (Segment1, Segment2, Label)
    # Berechnung der Wahrscheinlichkeit, dass Segment1 gegenüber Segment2 preferriert wird
    # Berechnung der Cross-Entropy-Verlustfunktion
    # Optimierung des Reward Models mittles Gradientenabstiegsverfahren
    def train_reward_model(self):
    
        if not self.labeled_data:
            print("Keine gelabelten Daten vorhanden. Training übersprungen.")
            return

        self.reward_model.train()  # Setze Reward Model in Trainingsmodus
        optimizer = self.reward_optimizer  # Verwende den Optimizer des PPO-Agenten

        epochs = 10  # Anzahl der Trainingsepochen
        batch_size = 32

        for epoch in range(epochs):
            random.shuffle(self.labeled_data)  # Zufällige Durchmischung der Daten
            for i in range(0, len(self.labeled_data), batch_size):
                # Labeled Data im Umfang von einem Batch aus labeled_data entnommen (in batch_size-Schritten)
                batch = self.labeled_data[i:i + batch_size]         

                # Batch-Daten extrahieren
                # Alle Segmente1, Segmente2 sowie alle Labels in separate Listen geschrieben
                inputs1, inputs2, labels = zip(*batch)  
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

                # Berechne die Rewards der beiden Segmente
                # Alle Segmente haben folgendes Format: (segment_obs_action, segment_true_reward, segment_predicted_reward)
                # Reward Model wird auf jedes Segment angewendet 
                # -> gibt für jedes Zustands-Aktions-Paar im Segment einen predicted reward
                rewards1 = [
                    torch.sum(      # durch torch.sum erhalten wir Gesamtreward
                        self.reward_model(torch.tensor(segment, dtype=torch.float32).to(self.device))
                    )
                    for segment in inputs1
                ]
                rewards2 = [
                    torch.sum(
                        self.reward_model(torch.tensor(segment, dtype=torch.float32).to(self.device))
                    )
                    for segment in inputs2
                ]

                # rewards1 und rewards2 sind Listen von Tensors
                # Kombiniert alle Tensors von rewards1 und rewards2 zu einem einzigen Tensor
                # Die resultierenden Tensors haben jeweils die From [batch_size]
                rewards1 = torch.stack(rewards1)  
                rewards2 = torch.stack(rewards2)  

                # Anwendungen der Formeln aus dem Christiano-Paper
                # Berechne \( P[\sigma^1 > \sigma^2] \)
                probabilities = torch.sigmoid(rewards1 - rewards2)

                # Berechne Cross-Entropy Loss
                loss = -torch.mean(
                    labels * torch.log(probabilities + 1e-8)
                    + (1 - labels) * torch.log(1 - probabilities + 1e-8)
                )

                # Backpropagation und Optimierung
                # Optimierung unseres Reward Models mittels Gradientenabstiegsverfahren
                optimizer.zero_grad()       # Gradienten aller Parameter des Modells auf 0 gesetzt (sonst würden sich Gradienten von mehreren Backward-Pässen aufaddieren)
                loss.backward()
                # Aktualisierung der Modellparameter basierend auf den mit loss.backward() berechneten Gradienten
                optimizer.step()

        # Nach dem Training die gelabelten Daten zurücksetzen
        self.labeled_data = []


        # Geändert, sodass für jedes Environment eine zusammenhängende Liste von Steps verwendet wird und daraus
        # zufällige Segmente mit je 60 Steps ausgewählt werden
        # Für jedes Environment separate Segmentliste erstellt -> Liste von Listen erstellt
    def select_segments(self):

        segments_per_env = [[] for _ in range(Args.num_envs)]  # Liste von Listen

        for env_id in range(Args.num_envs):
            # Extrahiere die Liste der Steps, True Rewards und Predicted Rewards für das Environment
            steps = self.obs_action_pair_buffer[env_id]
            true_rewards = self.true_reward_buffer[env_id]
            predicted_rewards = self.predicted_rewards_buffer[env_id]

            # Anzahl der Datenpunkte im aktuellen Environment
            num_steps = steps.shape[0]

            # Überprüfe, ob genügend Daten für Segmente vorhanden sind
            if num_steps < self.segment_size:
                continue

            # Wähle Segmente aus
            num_segments = num_steps // self.segment_size  # Maximale Anzahl von Segmenten
            for _ in range(num_segments):
                # Zufälligen Startindex wählen
                start_idx = np.random.randint(0, num_steps - self.segment_size)
                end_idx = start_idx + self.segment_size

                # Segmentdaten extrahieren
                segment_obs_action = steps[start_idx:end_idx]
                segment_true_reward = np.sum(true_rewards[start_idx:end_idx])
                segment_predicted_reward = np.sum(predicted_rewards[start_idx:end_idx])

                # Speichere das Segment
                segment = (segment_obs_action, segment_true_reward, segment_predicted_reward)
                segments_per_env[env_id].append(segment)

        # Speicherpuffer nach der Segmentauswahl zurücksetzen
        self.obs_action_pair_buffer = None
        self.true_reward_buffer = None
        self.predicted_rewards_buffer = None

        return segments_per_env  # Liste von Listen, eine pro Environment

        

    def label_segments(self):
    
        # Wählt zwei Segmente aus demselben Environment aus und erstellt ein Tripel:
        # (segment_1, segment_2, label), wobei label 1, 0 oder 0.5 ist, abhängig davon, welcher true reward größer ist.
        # Speichert die Tripel in der Datenbank labeled_data
        
        labeled_triplets = []  # Liste zum Speichern der gelabelten Tripel
        segments_per_env = self.select_segments()  # Liste von Listen mit Segmenten je Environment

        for env_id, segments in enumerate(segments_per_env):
            if len(segments) < 2:
                # Überspringe Environments mit weniger als zwei Segmenten
                continue

            # Paare von Segmenten innerhalb des gleichen Environments bilden
            for _ in range(len(segments) // 2):  # Begrenze Anzahl der Vergleiche
                segment_1 = segments.pop(random.randint(0, len(segments) - 1))
                segment_2 = segments.pop(random.randint(0, len(segments) - 1))

                true_reward_1 = segment_1[1]  # true reward von Segment 1
                true_reward_2 = segment_2[1]  # true reward von Segment 2

                # Label bestimmen
                if true_reward_1 > true_reward_2:
                    label = 1
                elif true_reward_1 < true_reward_2:
                    label = 0
                else:
                    label = 0.5

                # Tripel erstellen und speichern
                triplet = (segment_1, segment_2, label)
                labeled_triplets.append(triplet)

        # Speichere die gelabelten Daten in der Datenbank
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