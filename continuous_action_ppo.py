import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter




# Argumente Parsing 

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args



# Gym Umgebungs Erstellung 


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# Netzwerkinitialisierung 


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Agenten-Modell (neuronales Netzwerk) 

class Agent(nn.Module):

    """
    Der Agent besteht aus zwei Hauptteilen:
    - Einem Kritiker, der den Wert eines Zustands schätzt.
    - Einem Schauspieler, der die Wahrscheinlichkeitsverteilung über Aktionen erzeugt.
    """

    def __init__(self, envs):
        super(Agent, self).__init__()
        # Kritiker Netzwerk (Value Network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Schauspieler Netzwerk (Policy Network)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        
        # Log-Standardabweichung für kontinuierliche Aktionen
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None :
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x) 
    


# Hauptteil ------
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Wenn das Tracking aktiviert ist, wird Weights and Biases initialisiert
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=run_name)

    # Setze den Zufalls-Seed für Reproduzierbarkeit
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Erstelle die Umgebungen
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)])

    # Initialisiere das Agenten-Netzwerk
    agent = Agent(envs)
    agent.to(device=torch.device("cuda" if args.cuda else "cpu"))

    # Definiere den Optimierer
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialisiere TensorBoard-Writer, um Trainingsergebnisse zu speichern und zu visualisieren
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Füge die Hyperparameter als Text in TensorBoard hinzu
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Versuche, das Zufallssampling zu fixieren, indem du den Seed für alle relevanten Bibliotheken setzt
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic  # Ermöglicht deterministische Berechnungen auf CUDA, wenn aktiviert

    # Bestimme, ob CUDA (GPU) verfügbar ist und ob sie verwendet werden soll
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialisiere die Umgebung mit mehreren parallelen Umgebungen
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # Stelle sicher, dass die Umgebung den richtigen Aktionsraum hat (kontinuierlich)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Initialisiere den Agenten und den Optimierer
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Speicher für die beobachteten Daten und andere wichtige Variablen vorbereiten
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Versuche nicht zu ändern: Starte das Spiel mit einem Reset der Umgebung
    global_step = 0  # Die Anzahl der verarbeiteten Schritte
    start_time = time.time()  # Speichere die Zeit für die Berechnung der Schritte pro Sekunde (SPS)
    next_obs = torch.Tensor(envs.reset()).to(device)  # Setze die Umgebung zurück und speichere den ersten Zustand
    next_done = torch.zeros(args.num_envs).to(device)  # Initialisiere den "done"-Status (ob das Spiel für jede Umgebung beendet ist)
    num_updates = args.total_timesteps // args.batch_size  # Berechne die Anzahl der Updates basierend auf den Gesamt-Timesteps und Batch-Größe

    # Beginne das Training mit mehreren Updates
    for update in range(1, num_updates + 1):
        # Wenn das Lernen annealiert werden soll (abnehmende Lernrate), passe die Lernrate an
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates  # Berechne den aktuellen Fraktionswert
            lrnow = frac * args.learning_rate  # Berechne die neue Lernrate
            optimizer.param_groups[0]["lr"] = lrnow  # Setze die neue Lernrate im Optimierer

        # Iteriere über die Schritte innerhalb eines Rollouts
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs  # Erhöhe die Gesamtzahl der Schritte um die Anzahl der Umgebungen
            obs[step] = next_obs
            dones[step] = next_done

            # Berechne die Aktionen des Agenten
            with torch.no_grad():  # Keine Gradientenberechnung erforderlich
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()  # Speichere den Wert des aktuellen Zustands
            actions[step] = action
            logprobs[step] = logprob

            # Versuche nicht zu ändern: Führe die Aktion aus und erhalte das nächste Zustand, Belohnung und "done"-Status
            next_obs, reward, done, info = envs.step(action.cpu().numpy())  # Setze die Umgebung mit der Aktion fort
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # Speichere die Belohnungen für diesen Schritt
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)  # Update der nächsten Beobachtungen und "done"-Status

            # Wenn eine Episode abgeschlossen ist, logge die Belohnung und die Länge der Episode
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        
        
        # Bootstrap-Wert berechnen, falls die Episode nicht beendet ist
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)  # Berechne den Wert des nächsten Zustands
            if args.gae:
                # Berechne die Vorteile mit Generalized Advantage Estimation (GAE)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done  # Wenn die Episode beendet ist, wird nextnonterminal auf 0 gesetzt
                        nextvalues = next_value  # Der Wert des nächsten Zustands
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                # Berechne die Rückgaben ohne GAE
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        
        
        # Flache die Batch-Daten, um sie für das Update vorzubereiten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        
        
        # Optimierung der Politik- und Wertnetzwerke
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # Mische die Indizes der Batch-Daten
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Berechne neue Aktionen und Log-Wahrscheinlichkeiten
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Berechne den durchschnittlichen KL-Divergenzwert
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Normalisierung der Vorteile
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Berechne den Verlust für die Politik
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Berechne den Wertverlust
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

                # Entropieverlust berechnen
                entropy_loss = entropy.mean()

                # Gesamtverlust berechnen und den Optimierer aktualisieren
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # Gradientenbeschränkung
                optimizer.step()

            # Wenn der Ziel-KL-Wert überschritten wird, breche die Schleife ab
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        
        
        # Berechne den erklärten Varianz-Wert
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        
        
        # Speichere wichtige Werte wie Verlust und Lernrate für TensorBoard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        
        # Berechne und speichere die Schritte pro Sekunde
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    
    
    # Schließe die Umgebung und den TensorBoard-Writer nach dem Training
    envs.close()
    writer.close()
