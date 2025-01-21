import time
import torch
import numpy as np
import tyro
import os
import shutil
import threading
from rlhf.configs.arguments import Args
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import train_reward_model_ensemble
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

    total_queries = ppo.args.num_queries
    min_queries_per_training = 5
    amount_of_trainings = total_queries // min_queries_per_training
    div = num_iterations // amount_of_trainings
    queries_trained = 0

    # per_iter = total_queries // num_iterations
    # extra_at_start = total_queries % num_iterations

    data = []
    for iteration in range(1, num_iterations + 1):
        # queries = per_iter
        # if iteration == 1:
        #     queries += extra_at_start
        # queries_trained += queries

        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * ppo.args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

        ppo.collect_rollout_data()

        if iteration % div == 0:
            queries = min(min_queries_per_training, total_queries)

            if queries > 0:
                total_queries -= min_queries_per_training
                queries_trained += queries

                Labeling.counter = 0
                if args.synthetic == False:
                    global flask_port
                    if flask_port is None:  # Falls Flask noch nicht gestartet ist
                        flask_port = start_flask()

                labeling = Labeling(
                    segment_size,
                    ppo.args.synthetic,
                    ppo.args.uncertainty_based,
                    flask_port=flask_port,
                )
                labeled_data = labeling.get_labeled_data(
                    ppo.obs_action_pair_buffer,
                    ppo.env_reward_buffer,
                    ppo.predicted_rewards_buffer,
                    ppo.reward_models,
                    queries,
                    ppo.args.env_id,
                    iteration,
                )
                data.extend(labeled_data)

        if data:
            batch_size = 64
            tmp_data = data
            if len(data) > batch_size * 5:
                tmp_data = data[-batch_size * 5 :]
            train_reward_model_ensemble(
                ppo.reward_models, ppo.optimizers, tmp_data, ppo.device, batch_size
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

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    uploads_folder = os.path.join(BASE_DIR, "uploads")

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
