import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym  # OpenAI Gym für Umgebungen wie CartPole oder Atari
import numpy as np  # Für numerische Operationen
import torch  # Hauptbibliothek für maschinelles Lernen und Deep Learning
import torch.nn as nn  # Modul für neuronale Netzwerke
import torch.optim as optim  # Optimierungsalgorithmen wie Adam
from torch.distributions.categorical import Categorical  # Diskrete Wahrscheinlichkeitsverteilungen
from torch.utils.tensorboard import SummaryWriter  # Visualisierung von Trainingsdaten in TensorBoard



#Argumente für Skript definieren ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()  # Argument-Parser erstellen
    
    # Name des Experiments (Standard: Dateiname ohne .py)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    # Gym-Umgebungs-ID (z.B. CartPole-v1)
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    
    # Lernrate für den Optimizer (steuert, wie stark die Gewichte angepasst werden)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    
    # Zufalls-Seed für Reproduzierbarkeit
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    
    # Maximale Anzahl an Zeitschritten im Experiment
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    
    # Ob Torch deterministisch arbeiten soll (für reproduzierbare Ergebnisse)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    
    # Nutzung von CUDA für GPU-Beschleunigung
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    # Experiment-Tracking mit Weights and Biases
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    
    # Projektname in Weights and Biases
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    
    # Team-/Organisationsname in Weights and Biases
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    # Ob Videos von Agentenauftritten aufgenommen werden sollen
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")




# --- Algorithmusspezifische Argumente ----------------------------------------
    
    # Anzahl paralleler Spielumgebungen
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    
    # Anzahl Schritte pro Rollout (Batchgröße in Umgebungen)
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    
    # Annealing der Lernrate (lineare Reduktion während des Trainings)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    
    # GAE (Generalized Advantage Estimation) zur Berechnung von Vorteilen
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    
    # Discount-Faktor Gamma (Zukunftsbelohnung wie stark gewichtet wird)
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    
    # Lambda für GAE (reguliert Bias-Varianz-Abwägung)
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    
    # Anzahl der Mini-Batches
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    
    # Anzahl der Epochen pro Policy-Update
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    
    # Normalisierung der Vorteile (Advantage Values)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    
    # Clipping-Koeffizient für die Verlustfunktion
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    
    # Clipped Loss für Value Function
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    
    # Gewichtung der Entropie im Gesamtverlust (fördert Exploration)
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    
    # Gewichtung der Value-Funktion im Gesamtverlust
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    
    # Maximale Gradientennorm für Clipping
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    
    # Ziel-KL-Divergenz (stopt Training, wenn zu große Policy-Änderung)
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    



# --- Zusätzliche Berechnungen ------------------------------
    
    args = parser.parse_args()  # Parse-Befehl
    # Gesamtgröße eines Rollouts: Schritte x Umgebungen
    args.batch_size = int(args.num_envs * args.num_steps)
    # Mini-Batch-Größe: Batchgröße / Anzahl Mini-Batches
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args  # Gibt alle Argumente zurück




# Funktion: Erstellung von Gym-umgebungen

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        # Erstellt eine Gym-Umgebung basierend auf der ID
        env = gym.make(gym_id)

        # Setzt einen Seed für die Reproduzierbarkeit (z. B. gleiche Startbedingungen)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        # Falls Videos aktiviert sind, speichert die Umgebung Videos
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}",
                    episode_trigger=lambda ep: ep % 100 == 0  # Videoaufnahme alle 100 Episoden
                    )
        # Verpackt die Umgebung, sodass Observationen in Float32 skaliert werden
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env)
        return env

    return thunk




# Funktion: Initialisierung von Netzwerkgewichten ---------------------------------- 

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonale Initialisierung der Gewichte
    torch.nn.init.constant_(layer.bias, bias_const)  # Setzt die Bias auf eine Konstante
    return layer 



#Klasse Agent: Definition des neuronalen Netzwerks

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # --- Eingabedimensionen und Aktionsraum ---
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh()
            )
        
        #Policy-Kopf: Aktion auswählen
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        # Value-Kopf: Schätzt den Wert der aktuellen Observation
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, x):
        return self.network(x)

    def get_value(self, x):
        # Gibt den geschätzten Wert zurück
        return self.critic(self.forward(x))

    def get_action_and_value(self, x, action=None):
        # Berechnet die Policy (Aktion) und den geschätzten Wert
        hidden = self.forward(x)
        logits = self.actor(hidden)  # Aktion
        probs = Categorical(logits=logits)  # Wahrscheinlichkeitsverteilung

        if action is None:
            action = probs.sample()  # Zufällige Aktion basierend auf Policy

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)



# --- Hauptprogramm---------------------

if __name__ == "__main__":
    args = parse_args()  # Alle Argumente parsen

    # --- Setzen von Umgebungsvariablen -------------
        
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=run_name, save_code=True)

    # ---- TensorBoard initialisieren
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))

    # ---- Zufallszahlen für Reproduzierbarkeit setzen (try not to modify)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # --- Gerätewahl: GPU oder CPU --------------
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") 




# Erstellung paralleler Umgebungen 

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Nur diskrete Action Spaces werden unterstützt."

    agent = Agent(envs).to(device)  # Instanz des Agenten
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Adam-Optimizer



# ALGO logic: Storage SetUp
    # --- Rollout-Speicher vorbereiten ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialisierung der ersten Observationen
    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size





#Training Loop 

    for update in range(1, num_updates + 1):
        # Lernrate anpassen, falls aktiviert
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs  # Speichern der aktuellen Beobachtungen
            dones[step] = next_done

            # Agenten-Entscheidung: Aktion & Value
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Nächste Schritte ausführen
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.tensor(done).to(device)




# Vervollständigen der Rollouts (Ein Rollout ist eine Sequenz von Zuständen, Aktionen, Belohnungen und Übergängen, die ein Agent durch seine Interaktionen mit einer Umgebung generiert)

        # GAE-Vorteil und letzte Value-Berechnung
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = values[t + 1]
                delta = rewards[t] + args.gamma * nextnonterminal * next_return - values[t]
                advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae_lam

            returns = advantages + values




# Daten vorbereiten und mischen 

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Normalisierung der Vorteile
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)



# Optimierungsschleife- PPO 

        for epoch in range(args.update_epochs):
            data = MiniBatchSampler(b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, args)
            for batch in data:
                obs_batch, actions_batch, old_logprobs_batch, advantages_batch, returns_batch, values_batch = batch

                # Berechne neue Log-Wahrscheinlichkeiten und Werte
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_batch, actions_batch)
                logratio = newlogprob - old_logprobs_batch
                ratio = logratio.exp()

                # Clipped Objective
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value-Verlust
                value_loss = ((newvalue.flatten() - returns_batch) ** 2).mean()

                # Gesamter Verlust
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()

                # Rückwärts-Pass
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()




# Logging und Abschluss 


        # Logging der Ergebnisse
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)

        print(f"Update {update}/{num_updates} completed.")
