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
    def __init__(
        self, segment_size, synthetic, uncertainty_based, flask_port=None, test=False
    ):
        counter = 0
        self.segment_size = segment_size
        self.test = test
        self.synthetic = synthetic
        self.uncertainty_based = uncertainty_based
        self.flask_port = flask_port

    def preference_elicitation(self, segment_one, segment_two, env_id, iteration):
        """
        Vergleicht zwei Segmente und erstellt Labels für die Belohnungen.
        """
        segment_obs_actionOne, true_rewardOne, predicted_rewardOne = segment_one
        segment_obs_actionTwo, true_rewardTwo, predicted_rewardTwo = segment_two

        if self.synthetic:

            if true_rewardOne > true_rewardTwo:
                labelOne = 1
                labelTwo = 0
            elif true_rewardTwo > true_rewardOne:
                labelOne = 0
                labelTwo = 1
            else:
                labelOne = labelTwo = 0.5
            return (
                segment_obs_actionOne,
                segment_obs_actionTwo,
                (labelOne, labelTwo),
                (predicted_rewardOne, predicted_rewardTwo),
            )

        print("Labeling: Flask Port ist: ", self.flask_port)

        # Verzeichnisse flexibel erstellen
        # Basispfad wird dynamisch über abspath bestimmt, Ordner werden mit makedirs erstellt, falls sie nicht existieren
        # Dateien und Pfade werden mit path.join zusammengesetzt
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        upload_dir = os.path.join(base_dir, "uploads")
        base_url = f"http://127.0.0.1:{self.flask_port}"

        # Verzeichnisse erstellen, falls sie nicht existieren
        os.makedirs(upload_dir, exist_ok=True)

        # Verzeichnisse leeren
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
                    segment_obs_actionOne, _, predicted_rewardOne = segment_one
                    segment_obs_actionTwo, _, predicted_rewardTwo = segment_two
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

        return (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (predicted_rewardOne, predicted_rewardTwo),
        )

    def pairs_by_variance(self, segments, reward_models, queries: int):
        """
        Returns a list of tuples
        A tuple consists of (pair, variance) and the list is sorted by variance (in deacreasing order)
        A pair consists of two segments
        A segment consists of (segment_obs_action, true_reward, predicted_reward)
        """
        pairs_with_variance = []
        for _ in range(queries * 4):
            indices = np.random.choice(len(segments), 2, replace=False)
            pair = [segments[indices[0]], segments[indices[1]]]
            segment_obs_actionOne, _, _ = pair[0]
            segment_obs_actionTwo, _, _ = pair[1]
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
        predicted_rewards_buffer,
        reward_models,
        queries,
        env_id,
        iteration,
    ):
        """
        Vergleicht Segmente paarweise und erstellt die gelabelten Daten.
        """

        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, queries
        )

        labeled_data = []
        if self.uncertainty_based:

            pairs = self.pairs_by_variance(segments, reward_models, queries)
            labeled_data = []
            for segments, _ in pairs:
                segments_label_reward = self.preference_elicitation(
                    segments[0], segments[1], env_id, iteration
                )
                labeled_data.append(segments_label_reward)

            return labeled_data

        else:
            while len(segments) > 1 and queries > 0:
                queries -= 1
                segment_one = segments.pop()
                segment_two = segments.pop()
                segments_label_reward = self.preference_elicitation(
                    segment_one, segment_two, env_id, iteration
                )
                labeled_data.append(segments_label_reward)

            return labeled_data

    def select_segments(
        self,
        obs_action_pair_buffer,
        env_reward_buffer,
        predicted_rewards_buffer,
        queries,
    ):
        """
        Wählt zufällige Segmente aus den Buffern aus und berechnet deren Belohnungen.
        """
        obs_action_pair_buffer = np.array(obs_action_pair_buffer)
        env_reward_buffer = np.array(env_reward_buffer)
        predicted_rewards_buffer = np.array(predicted_rewards_buffer)

        data_points = len(env_reward_buffer)
        # achtung das muss noch geupdated werden, für varianz
        segment_amount = queries * 2

        segments = []
        for _ in range(segment_amount):
            start_idx = np.random.randint(0, data_points - self.segment_size)
            end_idx = start_idx + self.segment_size
            segment_obs_action = obs_action_pair_buffer[start_idx:end_idx]
            true_reward = sum(env_reward_buffer[start_idx:end_idx])
            predicted_reward = predicted_rewards_buffer[start_idx:end_idx]
            segment = (segment_obs_action, true_reward, predicted_reward)
            segments.append(segment)
        return segments
