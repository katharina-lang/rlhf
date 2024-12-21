import numpy as np
import os
import cv2


class Labeling:
    def __init__(self, segment_size=60, test=False, video_folder="segments_videos"):
        self.segment_size = segment_size
        self.test = test
        self.video_folder = video_folder
        os.makedirs(video_folder, exist_ok=True)


    def create_video_from_segment(self, frames, video_name, fps=30):
        """
        Erstellt ein Video aus den übergebenen Frames und speichert es.
        """
        height, width, _ = frames[0].shape
        video_path = os.path.join(self.video_folder, f"{video_name}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Konvertiere von RGB nach BGR für OpenCV
        out.release()

    def save_segment_videos(self, frames_buffer):
        """
        Speichert ein Video pro Segment. Jedes Segment besteht aus `segment_size` Frames.
        """
        total_frames = len(frames_buffer)
        num_segments = total_frames // self.segment_size

        for segment_idx in range(num_segments):
            start_idx = segment_idx * self.segment_size
            end_idx = start_idx + self.segment_size
            segment_frames = frames_buffer[start_idx:end_idx]
            video_name = f"segment_{segment_idx}"
            self.create_video_from_segment(segment_frames, video_name)
    

    def preference_elicitation(self, segment_one, segment_two):
        """
        Vergleicht zwei Segmente und erstellt Labels für die Belohnungen.
        """
        segment_obs_actionOne, true_rewardOne, predicted_rewardOne = segment_one
        segment_obs_actionTwo, true_rewardTwo, predicted_rewardTwo = segment_two

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
        self, obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer
    ):
        """
        Vergleicht Segmente paarweise und erstellt die gelabelten Daten.
        """
        labeled_data = []
        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer
        )

        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            segments_label_reward = self.preference_elicitation(
                segment_one, segment_two
            )
            labeled_data.append(segments_label_reward)

        return labeled_data
