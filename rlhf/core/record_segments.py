import gymnasium as gym
import os
import glob
import time
import imageio


def record_video_for_segment(env_id, segment, video_folder, segment_id, iteration):
    """
    Nimmt ein Video für ein bestimmtes Segment auf und speichert es in einem spezifischen Iterationsunterordner.
    """
    obs_action, _ = segment

    iteration_folder = os.path.join(video_folder)
    os.makedirs(iteration_folder, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()  # Setze die Umgebung zurück (obwohl wir die State-Änderung manuell steuern)

    frames = []
    obs, _ = env.reset()
    for i in range(len(obs_action)):
        obs = obs_action[i][: env.observation_space.shape[0]]
        action = obs_action[i][env.observation_space.shape[0] :]
        frame = env.render()
        frames.append(frame)

        # Manuelle Setzung des States in die Umgebung (Wichtig!)

        env.state = obs  # Setze den aktuellen Zustand der Umgebung
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    name_prefix = f"segment_{segment_id}_iteration_{iteration}.mp4"
    video_path = os.path.join(video_folder, name_prefix)
    imageio.mimsave(video_path, frames, fps=20)

    env.close()

    """
    # Umbenennen des Videos, um `episode_0` zu entfernen
    video_files = glob.glob(os.path.join(video_folder, f"segment_{segment_id}_iteration_{iteration}-episode-0.mp4"))
    for video_file in video_files:
        new_name = os.path.join(video_folder, f"segment_{segment_id}_iteration_{iteration}.mp4")
        os.rename(video_file, new_name)
        # Warte, bis die Datei sicher existiert
        while not os.path.exists(new_name):
            time.sleep(0.1)
    """
