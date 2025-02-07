import time
import torch
import numpy as np
import tyro
import os
import shutil
import random
from rlhf.configs.arguments import Args
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import train_reward_model_ensemble
from rlhf.core.labeling import Labeling
from rlhf.utils.app import start_flask, flask_port
from rlhf.core.record_segments import record_video_for_segment


def start_rollout_loop(ppo, num_iterations):
    """
    Starts the main rollout loop for training the agent and the reward model.

    Args:
        ppo (PPO): The PPO instance managing the agent and reward model training.
        num_iterations (int): Number of iterations to run the rollout loop.
    """
    segment_size = ppo.args.segment_size

    total_queries = ppo.args.num_queries
    queries_trained = 0
    queries_per_iter = max((total_queries // num_iterations) + 1, 3)

    train_data = []
    val_data = []
    for iteration in range(1, num_iterations + 1):

        if ppo.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * ppo.args.learning_rate
            ppo.optimizer.param_groups[0]["lr"] = lrnow

        ppo.collect_rollout_data()

        if not ppo.args.synthetic and iteration % 10 == 0:
            save_video(ppo.args.env_id, ppo.obs_action_pair_buffer, iteration)

        if total_queries > 0:
            queries = min(queries_per_iter, total_queries)

            total_queries -= queries
            queries_trained += queries

            if args.synthetic == False:
                global flask_port
                if flask_port is None:  # Falls Flask noch nicht gestartet ist
                    flask_port = start_flask()
                    print(
                        f"Running on http://127.0.0.1:{flask_port}/ (Press CTRL+C to quit)"
                    )

            labeling = Labeling(
                segment_size,
                ppo.args.synthetic,
                ppo.args.uncertainty_based,
                flask_port=flask_port,
            )
            labeled_data = labeling.get_labeled_data(
                ppo.obs_action_pair_buffer,
                ppo.env_reward_buffer,
                ppo.reward_models,
                queries,  # a query is the prompt for a pair of two trajectories
                ppo.args.env_id,
                iteration,
            )

            if ppo.args.validation:
                random.shuffle(labeled_data)
                split_idx = int(0.8 * len(labeled_data))
                train_data.extend(labeled_data[:split_idx])
                val_data.extend(labeled_data[split_idx:])
            else:
                random.shuffle(labeled_data)
                train_data.extend(labeled_data)

            batch_size = ppo.args.batch_size_rm

            dataset_size = 5
            if len(train_data) > batch_size * dataset_size:
                tmp_train_data = train_data[-batch_size * dataset_size :]
            else:
                tmp_train_data = train_data

            train_reward_model_ensemble(
                ppo.reward_models,
                ppo.optimizers,
                tmp_train_data,
                val_data,
                ppo.device,
                batch_size,
                writer=ppo.writer,
                global_step=ppo.global_step,
                anneal_dropout=ppo.args.anneal_dp,
                default_dropout=ppo.args.dropout,
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
    """
    Deletes all files and subdirectories within the specified folder.
    If the folder does not exist, it is created.

    Args:
        folder_path (str): The path to the folder to be cleared.

    Raises:
        Exception: If an error occurs while deleting files or directories.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Fehler beim Löschen von {file_path}: {e}")
    else:
        os.makedirs(folder_path)
        print(f"Directory {folder_path} was created.")


def save_video(env_id, obs_action, iteration):
    segment = (obs_action, 0)
    video_folder = f"segmentVideos_{env_id}"
    record_video_for_segment(env_id, segment, video_folder, 0, iteration)


if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    uploads_folder = os.path.join(BASE_DIR, "uploads")

    clear_uploads_folder(uploads_folder)

    ppo = PPO(run_name, args)

    if ppo.args.unsupervised_pretraining:
        print("Entered Unsupervised Pre-Training.")
        num_pt_iterations = int(0.01 * args.num_iterations)
        for pt_iteration in range(num_pt_iterations):
            ppo.collect_rollout_data(unsupervised_pretraining=True)
            avg_intrinsic_reward = torch.mean(ppo.rewards).item()
            print(
                f"Average Intrinsic Reward (Iteration {pt_iteration + 1}): {avg_intrinsic_reward}"
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
        print("Left Unsupervised Pre-Training.")

    # Start the rollout loop
    start_rollout_loop(ppo, args.num_iterations)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(ppo.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    ppo.envs.close()
    ppo.writer.close()
