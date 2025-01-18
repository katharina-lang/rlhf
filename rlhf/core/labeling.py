import numpy as np
import torch


class Labeling:
    def __init__(self, segment_size, synthethic, uncertainty_based, test=False):
        self.segment_size = segment_size
        self.test = test
        self.synthethic = synthethic
        self.uncertainty_based = uncertainty_based

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
        reward_models,
        queries,
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
                    segments[0], segments[1]
                )
                labeled_data.append(segments_label_reward)

            return labeled_data

        else:
            while len(segments) > 1 and queries > 0:
                queries -= 1
                segment_one = segments.pop()
                segment_two = segments.pop()
                segments_label_reward = self.preference_elicitation(
                    segment_one, segment_two
                )
                labeled_data.append(segments_label_reward)

            return labeled_data

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
