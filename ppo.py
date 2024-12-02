import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn   # Definition von neuronalen Netzwerken
import torch.optim as optim     # Definition von Optimierungsalgorithmen
from torch.distributions.categorical import Categorical     # Wahrscheinlichkeitsberechnungen von Aktionen
from torch.utils.tensorboard import SummaryWriter      # Zur Visualisierung von Trainingsdaten


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,      # Initialisierungswert für Zufallsgeneratoren für alle Pytorch-Zufallsoperationen, um Reproduzierbarkeit des Experiments zu gewährleisten
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,       # Gibt an, ob eine GPU für das Training verwendet werden soll, falls verfügbar
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,      # Anzahl paralleler Umgebungen, in denen der Agent gleichzeitig trainiert wird. Parallele Umgebungen beschleunigen das Sammeln von Erfahrungen.
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,   # Anzahl der Schritte pro Umgebung
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,      # Annealing der Learning Rate bedeutet, dass sie während des Trainings schrittweise reduziert wird
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,    # General Advantage Estimation: Wie viel besser ist eine bestimmte Aktion a in einem Zustand s, verglichen mit dem Durchschnittswert aller möglichen Aktionen in s
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,        # Discount-Faktor für zukünftige Belohnungen
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,   # Die Generalized Advantage Estimation (GAE) kombiniert kurzfristige und langfristige Vorteilsschätzungen mithilfe des Parameters Lambda: Bei kleinem Lambda gewichtet GAE nur den unmittelbaren TD-Fehler, bei großem Lambda gewichtet GAE zukünftige TD-Fehler stärker
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,   
        help="the number of mini-batches")
    # Minibatches sind kleine Teilmengen von Trainingsdaten, die in einer Optimierungsiteration verwendet werden
    # Vor Training: Trainingsdaten in Minibatches aufgeteilt
    # Training: Erfolgt in Epochen -> innerhalb jeder Epoche Minibatch aus Datensatz ausgewählt, 
    # Forward Pass der Minibatch-Daten durchs Modell, Vorhersage und Loss für jeden Minibatch berechnet,
    # Gesamtloss des Minibatches verwendet, Backward Pass, um festzustellen, wie die Gewichte geändert werden sollten 
    # (Gradienten zeigen nötige Anpassungen an), um den Loss zu minimieren
    # Gewichte entsprechend der Gradienten angepasst
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,   # Advantage-Werte können großen Schwankungen unterliegen -> Normalisierung sorgt für Stabilisierung
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,     # Clipping-Faktor zur Stabilisierung des PPO-Loss, d.h. um Änderungen der Policy von einer Version zur nächsten zu begrenzen
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,     # Clipping-Faktor für die Wertefunktion, die die erwartete kumulative Belohnung ab einem Zustand schätzt
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,     # Entropy coefficient -> Höhere Werte fördern exploratives Verhalten des Agents
        help="coefficient of the entropy")
    # Gesamtverlust in PPO setzt sich aus drei Komponenten zusammen:
    # 1. PPO Loss bzw. Policy Loss (Optimiert die Policy, um bessere Aktionen basierend auf den Advantage-Werten zu wählen)
    # 2. Value Loss (Optimiert die Wertefunktion, um zukünftige Belohnungen korrekt zu schätzen)
    # 3. Entropy Loss (Exploration angeregt)
    parser.add_argument("--vf-coef", type=float, default=0.5,   # Bestimmt, wie stark der Value Loss im Vergleich zum Policy-Loss gewichtet wird
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,     #B egrenzt die Größe der Gradienten beim Backward Pass -> Gewichte des Modells könnten sonst nach Anpassung sehr große Werte annehmen
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,    # Schwelle für die Kullback-Leibler (KL)-Divergenz -> sicherstellen, dass die neue Policy nicht zu stark von der vorherigen abweicht
        help="the target KL divergence threshold")
    # Unterschied zwischen target-kl und clip-coef:
    # clip-coef: Verhindert, dass sich die Wahrscheinlichkeit einer Aktion zwischen der alten und der neuen Policy drastisch verändert (Fokus auf einzelner Aktion)
    # target-kl: Betrachtet die gesamte Policy als Wahrscheinlichkeitsverteilung und stellt sicher, dass die Änderungen der Wahrscheinlichkeiten über alle Aktionen hinweg nicht zu groß sind
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)   # Gesamtanzahl der gesammelten Erfahrungen über alle Umgebungen und Zeitschritte
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# Beispiel: 100 parallele Umgebungen erstellen, um einen Reinforcement-Learning-Agenten zu trainieren
# Problem: Computer hat jedoch nur Ressourcen (z.B. CPU-Kerne), um beispielsweise 10 Umgebungen gleichzeitig auszuführen
# Jede Umgebung belegt Speicher, selbst wenn sie nicht aktiv genutzt wird -> 90 Umgebungen würden Speicher belegen, obwohl sie nicht aktiv sind -> Speicherüberlastung
# Für jede der 100 gewünschten Umgebungen wird eine Funktion thunk() erstellt
# thunk() ist keine Umgebung, sondern nur eine Beschreibung, wie die Umgebung erstellt wird
# Ein Parallelisierungswerkzeug (gym.vector.SyncVectorEnv) verwaltet diese Funktionen und startet nur so viele Umgebungen, wie die Ressourcen erlauben (10 in diesem Beispiel)
# Dadurch nur 10 Umgebungen gleichzeitig im Speicher
# Sobald Ressourcen frei werden, ruft SyncVectorEnv die nächste thunk()-Funktion auf, um eine neue Umgebung zu erstellen
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Standardabweichung auf sqrt(2) festgelegt
# bias_const legt den Startwert für die Biases des Layers fest, d.h. Grundaktivität (nicht-null Ausgabewert erzeugt, selbst wenn die Eingaben x gering oder null sind) eines Neurons am Anfang des Trainings auf 0
# orthogonal: Anwendung auf Gewichte W -> Spalten und Zeilen der Matrix sind linear unabhängig / orthogonal (d.h. W * W^T = Einheitsmatrix), was Korrelationen verhindert (Neuronen stören sich beim Lernprozess nicht gegenseitig)
# Nach Matrix-Erstellung wird jeder Wert in W mit std multipliziert
# constant: Anwendung auf Biases b -> Alle Werte gleich gesetzt (auf bias_const)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):       
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):       # Konstruktor -> beim Erstellen der Klasse automatisch aufgerufen
        super(Agent, self).__init__()       # Ruft den Konstruktor der Basisklasse (nn.Module) auf
        # In nn.Sequential werden die Layer wie eine Liste organisiert
        # Daten durchlaufen die Module in der Reihenfolge, in der sie in nn.Sequential definiert sind
        self.critic = nn.Sequential(        # Definiert das Critic-Netzwerk, das den Wert eines Zustands schätzt
            # nn.Linear: Linear-Layer (voll verbundene Layer) multipliziert Eingabe mit einer Gewichtsmatrix und addiert einen Bias
            # Eingabegröße: envs.single_observation_space.shape, d.h. die Dimension des Zustandsraums
            # Ausgabegröße: 64 Neuronen -> Eingabe s in einen 64-dimensionalen Raum projiziert
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            # Anwendung der Aktivierungsfunktion Tanh
            # Beschränkt Werte auf Bereich [-1,1]
            nn.Tanh(),
            # Eingabe: Die 64-dimensionalen Ausgaben der vorherigen Layer       
            # Eingabegröße: 64 Neuronen
            # Ausgabegröße: 64 Neuronen
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # Eingabe: 64 Neuronen aus der vorherigen Layer
            # Ausgabegröße: 1 Neuron -> geschätzter Wert des Eingabezustands
            # std=1.0 bedeutet, dass die Werte der Gewichtsmatrix mit 1 multipliziert werden, 
            # also die Werte aus der orthogonalen Initialisierung nicht verändert werden -> keine Skalierung
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # Ausgabe im Gegensatz zum Critic kein einzelner Wert, sondern ein Wahrscheinlichkeitsvektor 
            # über alle möglichen Aktionen
            # Kleinere initiale Gewichte (std=0.01 im Gegensatz zu std=1.0 bei Critic) sorgen für eine 
            # vorsichtige Startverteilung, bei der keine Aktion zu stark bevorzugt wird
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    # 
    def get_action_and_value(self, x, action=None):     # Optional kann eine spezifische Aktion übergeben werden -> Ansonsten wird eine aus der Wahrscheinlichkeitsverteilung gesampelt
        logits = self.actor(x)      # Logits: Rohe Ausgaben eines neuronalen Netzes, bevor sie in Wahrscheinlichkeiten umgewandelt werden
        probs = Categorical(logits=logits)      # Umwandlung Logits in Wahrscheinlichkeitsverteilung durch Softmax-Funktion
        if action is None:
            action = probs.sample()     # Beispiel: Aus Wahrscheinlichkeiten [0.71,0.04,0.25] wird die Aktion mit Wahrscheinlichkeit 71 %, 4 % oder 25 % gewählt
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # SummaryWriter: Erstellung von Logs für TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")      # Erstellt einen Ordner für das aktuelle Experiment mit dem Namen run_name
    writer.add_text(                                # Fügt eine Tabelle mit allen Hyperparametern hinzu, die in args definiert sind
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)      # Initialisiert den Zufallsgenerator des Python-Basismoduls random mit dem Seed args.seed
    np.random.seed(args.seed)   # Setzt Zufallsgenerator von NumPy auf den Seed args.seed
    torch.manual_seed(args.seed)        # Initialisiert den Zufallsgenerator von PyTorch mit dem Seed args.seed
    torch.backends.cudnn.deterministic = args.torch_deterministic       # Aktiviert (True) oder deaktiviert (False) deterministisches Verhalten für cuDNN, eine NVIDIA-Bibliothek zur Optimierung von Deep-Learning-Berechnungen auf GPUs

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # alle Umgebungen führen gleichzeitig einen Schritt aus und geben ihre Ergebnisse zurück -> Parallelität
    # make_env() wird für jede Umgebung aufgerufen -> thunk-Funktion zurückgegeben, die die Gym-Umgebung erstellt
    # Liste enthält Funktionen, die je nach Bedarf neue Umgebungen initialisieren können
    # Jede Umgebung erhält einen leicht unterschiedlichen Seed (z. B. 1, 2, 3 ...), um sicherzustellen, dass die Zustände der Umgebungen nicht identisch sind
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # Prüfen, ob Umgebungen mit diskretem Aktionsraum (kontinuierliche Aktionsräume nicht unterstützt)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    # Optimizer aktualisiert die Gewichte eines Modells basierend auf dem 
    # Gradienten der Verlustfunktion, um das Modell zu verbessern
    # Adam = Adaptive Moment Estimation
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # Erstellt leere Tensoren, die während Rollouts mit Daten gefüllt werden
    # Erstellt einen Tensor, der Beobachtungen (Zustände der Umgebung) speichert
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # Erstellt einen Tensor, der vom Agenten gespeicherte Zustände speichert
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # Erstellt einen Tensor, der die Log-Wahrscheinlichkeiten der gewählten Aktionen speichert
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Erstellt einen Tensor, der die Belohnungen speichert, die der Agent nach jeder Aktion erhält
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Erstellt einen Tensor, der speichert, ob eine Episode beendet wurde   
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Erstellt einen Tensor, der die geschätzten Werte des Critic-Modells speichert
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:      # Soll Lernrate während des Trainings schrittweise reduziert werden?
            frac = 1.0 - (update - 1.0) / num_updates       # Linearer Faktor frac, der von 1.0 (beim ersten Update) auf 0.0 (beim letzten Update) abnimmt
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow         # Aktualisiert die Lernrate des Optimierers Adam -> schrittweise reduziert

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs        # Globaler Schrittzähler um args.num_envs erhöht, denn mehrere Umgebungen parallel ausgeführt -> Ein Schritt in allen Umgebungen ergibt zusammen args.num_envs Schritte
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():   # Deaktiviert die automatische Gradientenberechnung in PyTorch, um Speicher zu sparen und die Berechnungsgeschwindigkeit zu erhöhen (möglich, da in diesem Schritt nur die Vorhersagen des Modells benötigt werden, aber keine Optimierung stattfindet)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()      # Geschätzte Werte für den aktuellen Zeitschritt gespeichert -> dabei flatten, zm Dimension des Tensors "value" zu reduzieren -> sicherzustellen, dass er die erwartete Form für weitere Berechnungen hat
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())      # Aktion in allen parallelen Umgebungen ausführen
            rewards[step] = torch.tensor(reward).to(device).view(-1)            # Konvertiert die Belohnungen aus der Umgebung (reward, anfänglich noch NumPy-Array) in einen PyTorch-Tensor -> mit view(-1) Formatierung in eindimensionale Struktur
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:       # Iteriert über die info-Objekte, die von jeder Gym-Umgebung zurückgegeben werden
                if "episode" in item.keys():      # Wenn Episode beendet: info-Objekt enthält Schlüssel "episode" mit zusätzlichen Statistiken
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)      # Summe der Belohnungen für die Episode
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)      # Länge der Episode
                    break

        # bootstrap value if not done -> falls Episode zu Ende: Nächster Zustandswert irrelevant
        with torch.no_grad():       # Deaktiviert Gradientenberechnung, da der next_value nur für die Berechnung von Rückgaben verwendet wird, nicht für das Training
            next_value = agent.get_value(next_obs).reshape(1, -1)       # Wert des nächsten Zustands durch Critic geschätzt -> reshape: eine Zeile und zweite Dimension automatisch berechnet, um alle verbleibenden Werte des Tensors abzudecken
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)       # Tensor erstellt, um Advantages zu speichern
                lastgaelam = 0                                          # Speichert letzten berechneten Generalized Advantage Estimation (GAE) Wert
                for t in reversed(range(args.num_steps)):               # Rückwärtsiteration von num_steps bis 0, denn discounted sum of rewards und advantages hängen von zukünftigen Werten ab
                    if t == args.num_steps - 1:                         
                        nextnonterminal = 1.0 - next_done               # Ist der nächste Zustand terminal?
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    # Berechnung des TD-Fehlers (Diskrepanz zwischen geschätztem Wert und tatsächlichem Wert)
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]      # Belohnung für aktuellen Schritt + diskontierte zukünftige Belohnung - Schätzung des aktuellen Zustandswertes
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam        # Berechnung des Generalized Advantage Estimation (GAE) bzw. des Advantage
                returns = advantages + values       # Rückgaben werden als Summe von Vorteilen und Zustandsschätzungen berechnet
            else:       # Ohne GAE wird der Advantage als Differenz zwischen den Rückgaben und den Zustandsschätzungen berechnet
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

        # flatten the batch
        # reshape ändert die Form eines Tensors, ohne dessen Inhalte zu verändern
        # Daten für die Minibatch-Verarbeitung vorbereiten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)     # Erstellt Liste von Indizes für den gesamten Batch
        clipfracs = []      # Speichert Anteil Minibatches, bei denen Änderung der Policy außerhalb der Range [1 - ϵ, 1 + ϵ] lag -> hoher Anteil an Clipping deutet darauf hin, dass die Policy zu schnell aktualisiert wird (schlechte Wahl der Lernrate oder des Clipping-Faktors ϵ)
        for epoch in range(args.update_epochs): # Iteration über Epochen, in denen die Policy und der Value-Loss pro Batch optimiert werden
            np.random.shuffle(b_inds)           # Reihenfolge der Indizes zufällig mischen -> sonst könnte Modell unnötige Abhängigkeiten von der Reihenfolge der Daten lernen
            for start in range(0, args.batch_size, args.minibatch_size):    # von 0 bis batch_size in minibatch_size-Schritten
                end = start + args.minibatch_size       # Berechnet Ende des aktuellen Minibatches
                mb_inds = b_inds[start:end]
                # Beispiel: args.batch_size=10, args.minibatch_size=3:
                # start=0, end=0+3=3
                # start=3, end=3+3=6
                # start=6, end=6+3=9
                # start=9, end=9+3=12 (endet hier, weil es über den Batch hinausgeht)
                # Für b_i​nds=[2,7,3,5,1,0,4,6,8,9] ergäbe das:
                # mb_inds=[2,7,3]
                # mb_inds=[5,1,0]
                # mb_inds=[4,6,8]
                # mb_inds=[9]

                # Berechnet Verhältnis der Wahrscheinlichkeiten zwischen der aktueller Policy und vorheriger Policy für die gleichen Aktionen
                # Benötigt, um PPO-Loss zu berechnen und zu starke Policy-Updates zu verhindern
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])        # Vorwärtsdurchlauf für Beobachtungen und Aktionen durchführen
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()      

                # Überprüfung, wie sich die Policy zwischen Iterationen verändert, und wie stark das Clipping genutzt wird
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()      # Vergleich zwischen alter und neuer Policy
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # Überprüft für jede Aktion, ob Policy-Änderung außerhalb des Clipping-Bereichs lag 
                    # Konvertiert Ergebnis in einen Float-Tensor, wobei 1 bedeutet, dass Clipping aktiv war, und 0, dass es nicht aktiv war
                    # Berechnet den Anteil der Datenpunkte, bei denen das Clipping aktiv war
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]     
                    
                # Advantages normalisieren
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss berechnet
                # Implementierung folgender Formel: L_policy​=min(r_t​(θ)⋅A_t​,clip(r_t​(θ),1−ϵ,1+ϵ)⋅A_t​)
                # Dabei: r_t​(θ)=(π_old​(a∣s) / π_new​(a∣s))
                pg_loss1 = -mb_advantages * ratio   # Hier Minus verwendet, damit später torch.max statt min aus der Formel verwendet werden kann
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # Misst, wie gut der Critic die erwarteten Rückgaben vorhersagt
                newvalue = newvalue.view(-1)        # newvalue repräsentiert vorhergesagten Werte des Critics für die Beobachtungen im Mini-Batch -> in eindimensionale Liste umgewandelt
                if args.clip_vloss:                 # Soll Value Loss geclippt werden?
                    # v_loss_unclipped und v_loss_clipped berechnen beide den quadratischen Fehler (MSE) zwischen den vom Critic geschätzten Werten und den Zielwerten, nur einmal wird noch Clipping angewendet
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(        # geclippter aktueller Wert V(s)
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)        # Wenn geclippter Wert zu stark von R(s) abweicht, wird ungeclippter Loss verwendet -> Sicherstellen, dass das Clipping die Optimierung nicht vollständig blockiert -> für jede Beobachtung im Minibatch
                    v_loss = 0.5 * v_loss_max.mean()        # durchschnittlicher Loss über gesamtes Minibatch
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Backpropagation und Aktualisierung der Modellparameter basierend auf dem Gradienten
                entropy_loss = entropy.mean()
                # Gesamter Loss aus den drei Komponenten Value Loss, Entropy Loss und Policy Loss berechnet
                # Dadurch können sowohl Actor als auch Critic gleichzeitig optimiert werden, während Exploration durch Entropie gefördert wird
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()       # Sicherstellen, dass alte Gradienten nicht versehentlich in den aktuellen Optimierungsschritt einfließen
                loss.backward()     # Berechnete Gradienten für jeden Parameter in den entsprechenden grad-Attributen der Parameterobjekte gespeichert
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)        # Norm der Gradienten auf maximalen Wert (args.max_grad_norm) begrenzt (Gradient Clipping) -> Training stabilisieren und Explodieren von Gradienten vermeiden
                optimizer.step()        # Optimierer aktualisiert die Parameter des Modells basierend auf den berechneten Gradienten

            if args.target_kl is not None:          # Wurde in Parametern KL-Schwellenwert definiert?
                if approx_kl > args.target_kl:      # Vergleicht die berechnete KL-Divergenz (approx_kl) mit dem definierten Schwellenwert (args.target_kl)
                    break                           # Zu großer KL-Wert: Neue Policy weicht zu stark von der alten ab -> Abbruch der Optimierung
        
        # b_values: Vorhergesagten Werte des Critics
        # b_returns: Die tatsächlichen Zielwerte des Mini-Batch
        # y_pred: Die vorhergesagten Werte des Critics
        # y_true: Die tatsächlichen Zielwerte (Rückgaben)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # Varianz der tatsächlichen Rückgabewerte
        var_y = np.var(y_true)
        # Erklärte Varianz ist ein Maß für die Qualität der Vorhersagen
        # explained_var=1: Perfekte Vorhersage
        # explained_var=0: Die Vorhersagen erklären nichts
        # explained_var<0: Die Vorhersagen sind schlechter als konstanter Schätzer (Mittelwert der Zielwerte) -> Vorhersagen nicht sinnvoll
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Loggt die wichtigsten Trainingsmetriken für die Analyse und Visualisierung im TensorBoard
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()