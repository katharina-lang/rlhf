import numpy as np

def create_test_data():
    obs_actions = [
        np.array([  # Environment 1
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [100, 101, 102, 103, 104],
            [200, 201, 202, 203, 204],
        ]),
        np.array([  # Environment 2
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [105, 106, 107, 108, 109],
            [205, 206, 207, 208, 209],
        ]),
    ]
    true_rewards = [
        np.array([0.01, 0.02, 7.3, 5.0]),
        np.array([0.03, 0.04, 1.1, 4.0]),
    ]
    predicted_rewards = [
        np.array([0.05, 0.06, 6.25, 2.5]),
        np.array([0.07, 0.08, 2.75, 1.5]),
    ]

    obs_actions_flat = np.vstack(obs_actions)
    true_rewards_flat = np.hstack(true_rewards).flatten()
    predicted_rewards_flat = np.hstack(predicted_rewards).flatten()

    return obs_actions_flat, true_rewards_flat, predicted_rewards_flat
