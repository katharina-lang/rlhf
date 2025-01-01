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

    counter = 0

    def __init__(self, segment_size=60, test=False):
        self.segment_size = segment_size
        self.test = test

    def preference_elicitation(self, segment_one, segment_two, env_id, iteration):
        """
        Vergleicht zwei Segmente und erstellt Labels für die Belohnungen.
        """
        # erstmal nicht nebenläufig: nimmt zwei Videos auf, verschiebt sie in den Ordner für Flask, labelt sie und löscht sie dann
        record_video_for_segment(env_id, segment_one, f"segment_videos", self.counter)
        self.counter += 1
        record_video_for_segment(env_id, segment_two, f"segment_videos", self.counter)
        self.counter += 1

        video_paths = []
        video_files = os.listdir('C:/users/hanna/rlhf/segment_videos')
        videos = [f for f in video_files if f.endswith('.mp4')]
        
        video_paths.append(f"C:/users/hanna/rlhf/segment_videos/{videos[0]}")
        video_paths.append(f"C:/users/hanna/rlhf/segment_videos/{videos[1]}")

        # verschieben
        for video in video_paths:
            shutil.move(video, 'C:/users/hanna/rlhf/rlhf/utils/static/uploads')

        while True:
                try:
                    while True:
                        # Button-Drücke bekommen
                        response = requests.get('http://127.0.0.1:5000/status')
                        state = response.json()
                        button_status = state['status']
                        response2 = requests.get('http://127.0.0.1:5000/set')
                        state2 = response2.json()
                        button_set = state2['set']
                        print(button_status)
                        print(button_set)
                        # falls Button gedrückt wurde, weitergehen, sonst darauf warten
                        if (button_set == True):
                            break
                        time.sleep(2)

                    if (button_set == True):
                        # labeln
                        segment_obs_actionOne, _, predicted_rewardOne = segment_one
                        segment_obs_actionTwo, _, predicted_rewardTwo = segment_two
                        labelOne, labelTwo = button_status

                        # Button auf ungedrückt setzen
                        button_set = False
                        response = requests.post('http://127.0.0.1:5000/set', json={"new_value": button_set})

                        print('label gesetzt', button_status)

                        # fertig verarbeitete Videos aus Ordner löschen
                        video_paths2 = []
                        video_files2 = os.listdir('C:/users/hanna/rlhf/rlhf/utils/static/uploads')
                        videos2 = [f2 for f2 in video_files2 if f2.endswith('.mp4')]
                        video_paths2.append(f"C:/users/hanna/rlhf/rlhf/utils/static/uploads/{videos2[0]}")
                        video_paths2.append(f"C:/users/hanna/rlhf/rlhf/utils/static/uploads/{videos2[1]}")

                        for video2 in video_paths2:
                            os.remove(video2)

                        break
                    time.sleep(1)

                except requests.exceptions.ConnectionError as e:
                    print(f"Fehler bei der Verbindung zum Flask-Server: {e}")
                    time.sleep(1)

        return (
            segment_obs_actionOne,
            segment_obs_actionTwo,
            (labelOne, labelTwo),
            (predicted_rewardOne, predicted_rewardTwo),
        )

    def select_segments(
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer
    ):
        """
        Wählt zufällige Segmente aus den Buffern aus und berechnet deren Belohnungen.
        """
        obs_action_pair_buffer = np.array(obs_action_pair_buffer)
        env_reward_buffer = np.array(env_reward_buffer)
        predicted_rewards_buffer = np.array(predicted_rewards_buffer)

        data_points = len(env_reward_buffer)
        segment_amount = data_points // self.segment_size

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

    def get_labeled_data(
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, env_id, iteration
    ):
        """
        Vergleicht Segmente paarweise und erstellt die gelabelten Daten.
        """
        labeled_data = []
        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer
        )

        # Flask als Thread starten
        flask_thread = Thread(target=start_flask)
        flask_thread.start()

        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            segments_label_reward = self.preference_elicitation(
                segment_one, segment_two, env_id, iteration
            )
            labeled_data.append(segments_label_reward)
        
        flask_thread.join

        return labeled_data
