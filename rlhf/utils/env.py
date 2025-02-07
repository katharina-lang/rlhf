import gymnasium as gym
import numpy as np


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=None)
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
            )
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk