import gymnasium as gym
import os
import glob
import time

def record_video_for_segment(env_id, segment, video_folder, segment_id, iteration):
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
        name_prefix=f"segment_{segment_id}_iteration_{iteration}"  # Eindeutiger Name für das Video
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
    

    
    
