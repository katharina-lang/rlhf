import numpy as np
import torch
import os
import shutil
import requests
import time
from threading import Thread
from rlhf.utils.app import start_flask
from rlhf.core.record_segments import record_video_for_segment


class Labeling:
    """
    Handles the process of labeling trajectory segments for reward model training

    This class manages preference elicitation, segment selection and uncertainty-based labeling,
    integrating a Flask-based interface for human feedback.
    """

    def __init__(
        self, segment_size, synthetic, uncertainty_based, flask_port=None, test=False
    ):
        self.counter = 0
        self.segment_size = segment_size
        self.test = test
        self.synthetic = synthetic
        self.uncertainty_based = uncertainty_based
        self.flask_port = flask_port

    def preference_elicitation(self, segment_one, segment_two, env_id, iteration):
        """
        Compares two trajectory segments and assigns preference labels.

        If `synthetic` is True, preference labels are assigned based on reward values.
        Otherwise, a Flask-based interface is used for human preference elicitation.

        Args:
            segment_one (tuple): First segment containing (obs_action_pairs, true_reward).
            segment_two (tuple): Second segment containing (obs_action_pairs, true_reward).
            env_id (str): Environment identifier.
            iteration (int): Training iteration index.

        Returns:
            tuple: (obs_action_pairs_1, obs_action_pairs_2, (label_1, label_2))
        """
        segment_obs_actionOne, true_rewardOne = segment_one
        segment_obs_actionTwo, true_rewardTwo = segment_two

        if self.synthetic:

            if true_rewardOne > true_rewardTwo:
                labelOne = 1
                labelTwo = 0
            elif true_rewardTwo > true_rewardOne:
                labelOne = 0
                labelTwo = 1
            else:
                labelOne = labelTwo = 0.5
            return (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo))


        # Verzeichnisse flexibel erstellen
        # Basispfad wird dynamisch über abspath bestimmt, Ordner werden mit makedirs erstellt, falls sie nicht existieren
        # Dateien und Pfade werden mit path.join zusammengesetzt
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        upload_dir = os.path.join(base_dir, "uploads")
        base_url = f"http://127.0.0.1:{self.flask_port}"

        os.makedirs(upload_dir, exist_ok=True)

        def clear_directory(directory):
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # Dateien löschen
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Unterverzeichnisse löschen
                    except Exception as e:
                        print(f"Fehler beim Löschen von {file_path}: {e}")

        clear_directory(upload_dir)

        record_video_for_segment(
            env_id, segment_one, upload_dir, self.counter, iteration
        )
        self.counter += 1
        record_video_for_segment(
            env_id, segment_two, upload_dir, self.counter, iteration
        )
        self.counter += 1

        videos_ready = True
        # hier wird jetzt an Flask das Signal gesendet, videos_ready auf True zu setzen, da Videos fertig aufgenommen sind -> können angezeigt werden
        response = requests.post(
            f"{base_url}/setVideosReady", json={"new_value": videos_ready}
        )

        # Äußere Schleife sorgt für eine Wiederholung, falls es beim Abruf der Serverdaten zu einem Fehler kommt
        while True:
            try:
                # innere Schleife wartet explizit auf einen Button-Status, um Interaktion zu erkennen
                while True:
                    response = requests.get(f"{base_url}/status")
                    state = response.json()
                    button_status = state["status"]

                    response2 = requests.get(f"{base_url}/set")
                    state2 = response2.json()
                    button_set = state2["set"]

                    # falls Button gedrückt wurde, weitergehen, sonst darauf warten
                    if button_set:
                        break
                    time.sleep(0.1)

                if button_set:
                    segment_obs_actionOne, _ = segment_one
                    segment_obs_actionTwo, _ = segment_two
                    labelOne, labelTwo = button_status

                    button_set = False
                    # hier wird jetzt an Flask das Signal gesendet, den Button-Set wieder auf False zu setzen
                    response = requests.post(
                        f"{base_url}/set", json={"new_value": button_set}
                    )

                    break
                time.sleep(0.1)

            except requests.exceptions.ConnectionError as e:
                print(f"Fehler bei der Verbindung zum Flask-Server: {e}")
                time.sleep(0.1)

        return (segment_obs_actionOne, segment_obs_actionTwo, (labelOne, labelTwo))

    def pairs_by_variance(self, segments, reward_models, queries: int):
        """
        Selects segment pairs based on reward model prediction variance.

        Args:
            segments (list): List of available segments.
            reward_models (list): List of reward models.
            queries (int): Number of segment pairs to return.

        Returns:
            list: List of (segment_pair, variance), sorted in descending order of variance.
        """
        pairs_with_variance = []
        for _ in range(queries * 4):
            indices = np.random.choice(len(segments), 2, replace=False)
            pair = [segments[indices[0]], segments[indices[1]]]
            segment_obs_actionOne, _ = pair[0]
            segment_obs_actionTwo, _ = pair[1]
            segment_obs_actionOne = torch.tensor(segment_obs_actionOne)
            segment_obs_actionTwo = torch.tensor(segment_obs_actionTwo)
            choices = np.zeros(len(reward_models))

            for i, model in enumerate(reward_models):
                with torch.no_grad():
                    pred_r1 = model(segment_obs_actionOne).sum()
                    pred_r2 = model(segment_obs_actionTwo).sum()
                if pred_r1 > pred_r2:
                    choices[i] = 1
                elif pred_r2 > pred_r1:
                    choices[i] = 2
                else:
                    choices[i] = 0
            variance = np.var(choices)
            pairs_with_variance.append((pair, variance))
        pairs_with_variance.sort(key=lambda x: x[1], reverse=True)
        return pairs_with_variance[:queries]

    def get_labeled_data(
        self,
        obs_action_pair_buffer,
        env_reward_buffer,
        reward_models,
        queries,
        env_id,
        iteration,
    ):
        """
        Labels trajectory segments based on human or synthetic preference elicitation.

        Args:
            obs_action_pair_buffer (list): Buffer of observation-action pairs.
            env_reward_buffer (list): Buffer of environment rewards.
            reward_models (list): List of trained reward models.
            queries (int): Number of segment pairs to label.
            env_id (str): Environment identifier.
            iteration (int): Training iteration index.

        Returns:
            list: Labeled trajectory triplets.
        """

        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, queries
        )

        labeled_data = []
        if self.uncertainty_based:

            pairs = self.pairs_by_variance(segments, reward_models, queries)
            labeled_data = []
            for segments, _ in pairs:
                labeled_triplet = self.preference_elicitation(
                    segments[0], segments[1], env_id, iteration
                )
                labeled_data.append(labeled_triplet)

            return labeled_data

        else:
            while len(segments) > 1 and queries > 0:
                queries -= 1
                segment_one = segments.pop()
                segment_two = segments.pop()
                labeled_triplet = self.preference_elicitation(
                    segment_one, segment_two, env_id, iteration
                )
                labeled_data.append(labeled_triplet)

            return labeled_data

    def select_segments(
        self,
        obs_action_pair_buffer,
        env_reward_buffer,
        queries,
    ):
        """
        Selects random trajectory segments from the buffer.

        Args:
            obs_action_pair_buffer (list): Buffer of observation-action pairs.
            env_reward_buffer (list): Buffer of environment rewards.
            queries (int): Number of segment pairs to extract.

        Returns:
            list: Selected trajectory segments.
        """
        obs_action_pair_buffer = np.array(obs_action_pair_buffer)
        env_reward_buffer = np.array(env_reward_buffer)

        data_points = len(env_reward_buffer)
        # doppelt so viele Segmente wie queries, da zu jedem labeled_pair zwei segmente gehören
        segment_amount = queries * 2

        segments = []
        for _ in range(segment_amount):
            start_idx = np.random.randint(0, data_points - self.segment_size)
            end_idx = start_idx + self.segment_size
            segment_obs_action = obs_action_pair_buffer[start_idx:end_idx]
            env_reward = sum(env_reward_buffer[start_idx:end_idx])
            segment = (segment_obs_action, env_reward)

            segments.append(segment)
        return segments
