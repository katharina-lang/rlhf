import gymnasium as gym
import os
import imageio


def record_video_for_segment(env_id, segment, video_folder, segment_id, iteration):
    """
    Records a video for a specific segment and saves it in a designated iteration subfolder.

    Args:
        env_id (str): The ID of the Gym environment to be used for rendering.
        segment (tuple): A tuple containing the observation-action pairs for the segment.
        video_folder (str): The directory where the recorded video will be stored.
        segment_id (int): The identifier for the segment being recorded.
        iteration (int): The current iteration of the recording process.
    """
    obs_action, _ = segment

    iteration_folder = os.path.join(video_folder)
    os.makedirs(iteration_folder, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    frames = []
    obs, _ = env.reset()
    for i in range(len(obs_action)):
        obs = obs_action[i][: env.observation_space.shape[0]]
        action = obs_action[i][env.observation_space.shape[0] :]
        frame = env.render()
        frames.append(frame)

        env.state = obs
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    name_prefix = f"segment_{segment_id}_iteration_{iteration}.mp4"
    video_path = os.path.join(video_folder, name_prefix)
    imageio.mimsave(video_path, frames, fps=20)

    env.close()
