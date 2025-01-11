import numpy as np


class Labeling:
    def __init__(self, segment_size=60, test=False):
        self.segment_size = segment_size
        self.test = test

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

    def get_labeled_data(
        self,
        obs_action_pair_buffer,
        env_reward_buffer,
        predicted_rewards_buffer,
        queries,
    ):
        """
        Vergleicht Segmente paarweise und erstellt die gelabelten Daten.
        """
        labeled_data = []
        segments = self.select_segments(
            obs_action_pair_buffer, env_reward_buffer, predicted_rewards_buffer, queries
        )

        while len(segments) > 1:
            segment_one = segments.pop()
            segment_two = segments.pop()
            segments_label_reward = self.preference_elicitation(
                segment_one, segment_two
            )
            labeled_data.append(segments_label_reward)

        return labeled_data
