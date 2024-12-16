import torch
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import RewardModel
from rlhf.testAgent.test_args import TestArgs
from rlhf.testAgent.create_test_data import create_test_data


def test_agent():
    # Initialize Arguments and PPO Agent
    args = TestArgs()
    reward_model = RewardModel(input_dim=5)
    
    # Load Test Data
    obs_actions, true_rewards, predicted_rewards = create_test_data()
    test_data = (obs_actions, true_rewards, predicted_rewards)  # Testdaten als Tuple
    
    # Erstelle PPO-Agent mit Testdaten
    ppo_agent = PPO("test_run", args, test_data=test_data)

    # Perform Segment Selection and Labeling
    segments = ppo_agent.select_segments()
    ppo_agent.labeling.label_segments(ppo_agent)

    # Output Results
    print("=== Test Results ===")
    print("Segments:", segments)
    print("Labeled Data:", ppo_agent.labeled_data)
