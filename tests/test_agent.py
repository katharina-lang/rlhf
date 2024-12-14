import numpy as np
import torch
from rlhf.core.agent import Agent
from rlhf.core.reward_model import RewardModel
from rlhf.core.ppo import PPO
from rlhf.configs.arguments import Args

# Suppress Deprecation Warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Dummy Arguments for PPO
class TestArgs:
    num_envs = 2
    num_steps = 4
    segment_size = 2
    gamma = 0.99
    gae_lambda = 0.95
    seed = 42
    learning_rate = 0.001
    total_timesteps = 1000
    num_minibatches = 4
    torch_deterministic = True
    clip_coef = 0.2
    env_id = "HalfCheetah-v5"
    exp_name = "TestRun"
    capture_video = True
    cuda = False

# Create Test Data
def create_test_data():
    obs_actions = [
        np.array([  # Environment 1
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [100, 101, 102, 103, 104],
            [200, 201, 202, 203, 204],  # Add more steps
        ]),
        np.array([  # Environment 2
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [105, 106, 107, 108, 109],
            [205, 206, 207, 208, 209],  # Add more steps
        ]),
    ]
    true_rewards = [
        np.array([0.01, 0.02, 7.3, 5.0]),  # Add more rewards
        np.array([0.03, 0.04, 1.1, 4.0]),  # Add more rewards
    ]
    predicted_rewards = [
        np.array([0.05, 0.06, 6.25, 2.5]),  # Add more predicted rewards
        np.array([0.07, 0.08, 2.75, 1.5]),  # Add more predicted rewards
    ]
    return obs_actions, true_rewards, predicted_rewards


# Test the PPO Agent
def test_agent():
    args = TestArgs()
    reward_model = RewardModel(input_dim=5)
    ppo_agent = PPO("test_run", args, reward_model=reward_model)

    # Set Test Data
    obs_actions, true_rewards, predicted_rewards = create_test_data()
    ppo_agent.obs_action_pair_buffer = obs_actions
    ppo_agent.true_reward_buffer = true_rewards
    ppo_agent.predicted_rewards_buffer = predicted_rewards

    # Perform Segment Selection and Labeling
    segments = ppo_agent.select_segments()
    ppo_agent.label_segments()

    print("=== Test Results ===")
    print("Segments:", segments)
    print("Labeled Data:", ppo_agent.labeled_data)

if __name__ == "__main__":
    test_agent()
