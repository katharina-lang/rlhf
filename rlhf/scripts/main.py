import time
import torch
import numpy as np
import tyro
import os
import shutil
import threading
from rlhf.configs.arguments import Args
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import train_reward_model, train_reward_model_ensemble
from rlhf.core.labeling import Labeling
from rlhf.utils.app import start_flask, flask_port, monitor_app

def start_rollout_loop(ppo, num_iterations):
    """
    Starts the main rollout loop for training the agent and the reward model.

    Parameters:
        ppo (PPO): The PPO instance managing the agent and reward model training.
        num_iterations (int): Number of iterations to run the rollout loop.
    """
    segment_size = 60

    for iteration in range(1, num_iterations + 1):
        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * ppo.args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

        ppo.collect_rollout_data()
        
        Labeling.counter = 0
        preferences_for_iteration = calculate_preferences(
            iteration, num_iterations, ppo.args.amount_preferences
        )
        global flask_port
        if flask_port is None:  # Falls Flask noch nicht gestartet ist
            flask_port = start_flask()
        labeling = Labeling(segment_size=args.segment_size, test=False, flask_port=flask_port)
        labeled_data = labeling.get_labeled_data(
            ppo.obs_action_pair_buffer, 
            ppo.env_reward_buffer, 
            ppo.predicted_rewards_buffer, 
            ppo.args.env_id, 
            iteration,  # Übergibt die Iteration
            preferences_for_iteration
        )
        # Assign labeled data to the PPO agent
        ppo.labeled_data = labeled_data

        train_reward_model_ensemble(
            ppo.reward_models, ppo.optimizers, labeled_data, ppo.device
        )
        
        ppo.advantage_calculation()

        ppo.agent.optimize_agent_and_critic(
            ppo.obs,
            ppo.actions,
            ppo.logprobs,
            ppo.advantages,
            ppo.returns,
            ppo.values,
            ppo.optimizer,
            ppo.args,
        )

        # Calculate explained variance for debugging purposes
        y_pred, y_true = ppo.values.cpu().numpy(), ppo.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ppo.record_rewards_for_plotting_purposes(explained_var)

def calculate_preferences(iteration, total_iterations, total_preferences):
    """
    Berechnet die Anzahl der Präferenzen, die in der aktuellen Iteration abgefragt werden sollen.
    Die Verteilung erfolgt nichtlinear, z.B. durch exponentielle Abnahme.
    
    Args:
        iteration (int): Aktuelle Iteration.
        total_iterations (int): Gesamte Anzahl an Iterationen.
        total_preferences (int): Gesamte Anzahl an Präferenzen.

    Returns:
        int: Anzahl der Präferenzen für diese Iteration.
    """
    # Parameter für die exponentielle Abnahme
    alpha = 3  # Steuerung der Abnahmegeschwindigkeit
    weight = np.exp(-alpha * (iteration / total_iterations))
    preferences_for_iteration = int(total_preferences * weight)

    # Mindestanzahl an Präferenzen sichern (z.B. 1 Präferenz pro Iteration)
    return max(preferences_for_iteration, 1)

def clear_uploads_folder(folder_path):
    """Bereinigt den Ordner `uploads`, indem alle Dateien und Unterordner gelöscht werden."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Datei oder Symlink löschen
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Unterordner löschen
            except Exception as e:
                print(f"Fehler beim Löschen von {file_path}: {e}")
    else:
        # Falls der Ordner nicht existiert, erstelle ihn
        os.makedirs(folder_path)
        print(f"Ordner {folder_path} wurde erstellt.")

if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    uploads_folder = os.path.join(BASE_DIR, 'uploads')

    # Ordner bereinigen vor dem Start von Flask
    clear_uploads_folder(uploads_folder)

    ppo = PPO(run_name, args, test_data=False)
    # Start the rollout loop
    start_rollout_loop(ppo, args.num_iterations)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(ppo.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    ppo.envs.close()
    ppo.writer.close()
