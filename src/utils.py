import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.reward_model import reward_model_loss
from torch.distributions.normal import Normal
import random
import cv2
from torch.utils.tensorboard import SummaryWriter
from src.reward_model import RewardModel, RewardDataset, reward_model_loss
from src.init_arguments import Args
import time


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=gym.spaces.Box(
                low=-10,
                high=10,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            ),
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def generate_pairwise_data(trajectory_segments, num_pairs=5):
    pairwise_data = []
    for _ in range(min(num_pairs, len(trajectory_segments) // 2)):
        seg1, seg2 = random.sample(trajectory_segments, 2)
        score1 = sum(step["reward"] for step in seg1)
        score2 = sum(step["reward"] for step in seg2)
        preference = 1 if score1 > score2 else 0
        pairwise_data.append((seg1, seg2, preference))
    return pairwise_data


def save_segment_video(segment, env_id, filename, fps=30):
    """
    Save a trajectory segment as a video by resetting the environment
    and replaying the trajectory.
    """
    env = gym.make(
        env_id, render_mode="rgb_array"
    )  # Create a new instance for rendering
    frames = []
    env.reset()
    for step in segment:
        obs = step["state"]
        action = step["action"]

        # Render frame
        frame = env.render()
        frames.append(frame)

        # Step in the environment (using action to simulate trajectory)
        env.step(action)

    env.close()

    # Save frames as a video
    import imageio.v2 as imageio

    imageio.mimsave(filename, frames, fps=10)


def show_videos(video_files):
    """
    Show two videos side-by-side and get feedback.
    """
    cap1 = cv2.VideoCapture(video_files[0])
    cap2 = cv2.VideoCapture(video_files[1])
    key = -1  # Initialize key variable

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # If either video ends, stop playing both
        if not ret1 or not ret2:
            break

        # Combine the frames side-by-side
        combined_frame = cv2.hconcat([frame1, frame2])
        cv2.imshow("Choose: [1] Left [2] Right [0] Disregard", combined_frame)

        # Use a short delay between frames, allowing the video to play
        key = cv2.waitKey(30)  # Wait for 30ms for a key press
        if key in [49, 50, 48]:  # Valid keys: 1, 2, or 0
            break

    # Release the video capture objects
    cap1.release()
    cap2.release()

    # Wait for user input if no valid key was pressed during playback
    if key not in [49, 50, 48]:
        print("Playback finished. Waiting for user input...")
        while True:
            # Keep the last frame visible and wait indefinitely for valid input
            cv2.imshow("Choose: [1] Left [2] Right [0] Disregard", combined_frame)
            key = cv2.waitKey(0)  # Wait indefinitely for valid key
            if key in [49, 50, 48]:
                break

    cv2.destroyAllWindows()  # Close the video window only after user input
    return key  # Return user input (49 -> 1, 50 -> 2, 48 -> 0)


def show_and_get_feedback(segment1, segment2, env):
    """
    Display two trajectory segments to the human and get feedback.
    """
    # Save the segments as videos
    save_segment_video(segment1, env, "segment_0.mp4")
    save_segment_video(segment2, env, "segment_1.mp4")

    # Show the videos to the user
    feedback = show_videos(["segment_0.mp4", "segment_1.mp4"])

    # Interpret feedback
    if feedback == 49:  # Left video preferred
        return (segment1, segment2, 1)
    elif feedback == 50:  # Right video preferred
        return (segment1, segment2, 0)
    elif feedback == 48:  # Disregard both
        return None  # Skip this comparison
