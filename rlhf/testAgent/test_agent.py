import torch
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import RewardModel
from rlhf.testAgent.test_args import TestArgs
from rlhf.testAgent.create_test_data import create_test_data


def test_agent():
    # Initialize Arguments and PPO Agent
    args = TestArgs()
    reward_model = RewardModel(input_dim=5)
    ppo_agent = PPO("test_run", args, reward_model=reward_model)

    # Load Test Data
    obs_actions, true_rewards, predicted_rewards = create_test_data()
    ppo_agent.obs_action_pair_buffer = obs_actions
    ppo_agent.true_reward_buffer = true_rewards
    ppo_agent.predicted_rewards_buffer = predicted_rewards

    # Perform Segment Selection and Labeling
    segments = ppo_agent.select_segments()
    ppo_agent.label_segments()

    # Output Results
    print("=== Test Results ===")
    print("Segments:", segments)
    print("Labeled Data:", ppo_agent.labeled_data)
