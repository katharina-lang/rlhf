import gymnasium as gym
import os

def record_video_for_segment(env_id, segment, video_folder, segment_id):
    """
    Nimmt ein Video für ein bestimmtes Segment auf und speichert es in einem spezifischen Iterationsunterordner.
    """
    obs_action, _, _ = segment

    # Erstelle einen Unterordner für die aktuelle Iteration
    iteration_folder = os.path.join(video_folder)
    os.makedirs(iteration_folder, exist_ok=True)

    # Passe den RecordVideo-Wrapper an, um Videos eindeutig zu benennen
    env = gym.wrappers.RecordVideo(
        gym.make(env_id, render_mode='rgb_array'),
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix=f"segment_{segment_id}"  # Eindeutiger Name für das Video
    )

    env.reset()  # Setze die Umgebung zurück (obwohl wir die State-Änderung manuell steuern)
    
    for i in range(len(obs_action)):
        obs = obs_action[i][:env.observation_space.shape[0]]
        action = obs_action[i][env.observation_space.shape[0]:]
        
        # Manuelle Setzung des States in die Umgebung (Wichtig!)
        env.state = obs  # Setze den aktuellen Zustand der Umgebung (bei CartPole ist `state` ein Array)
        
        # Schritte ausführen (mit gespeicherter Action)
        obs, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            break

    env.close()
