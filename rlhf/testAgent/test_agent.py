import torch
from rlhf.core.ppo import PPO
from rlhf.core.reward_model import RewardModel
from rlhf.core.labeling import Labeling
from rlhf.testAgent.test_args import TestArgs
from rlhf.testAgent.create_test_data import create_test_data


def test_agent():
    # Initialize Arguments and PPO Agent
    args = TestArgs()
    reward_model = RewardModel(input_dim=5)
    
    # Load Test Data
    obs_actions, true_rewards, predicted_rewards = create_test_data()
    print(obs_actions)
    print(true_rewards)
    print(predicted_rewards)
    test_data = (obs_actions, true_rewards, predicted_rewards)  # Testdaten als Tuple
    
    # Erstelle PPO-Agent mit Testdaten
    ppo_agent = PPO("test_run", args, test_data=test_data)

    # Perform Segment Selection and Labeling
    labeling = Labeling(segment_size=2, test=True)  # Smaller segments for testing
    segments = labeling.select_segments(obs_actions, true_rewards, predicted_rewards)
    labeled_data = labeling.get_labeled_data(obs_actions, true_rewards, predicted_rewards)

    # Assign labeled data to the PPO agent
    ppo_agent.labeled_data = labeled_data

    # Output Results
    print("=== Test Results ===")
    print("Segments:", segments)
    print("Labeled Data:", ppo_agent.labeled_data)
