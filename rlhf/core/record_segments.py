import gymnasium as gym 

def record_video_for_segment(env_id, segment, video_folder):
        
        obs_action, true_reward, predicted_reward = segment
        env = gym.make(env_id, render_mode='rgb_array')
        obs_length = env.observation_space.shape[0]
        action_length = env.action_space.shape[0]
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: True)

        env.reset()  # Setze die Umgebung zurück (obwohl wir die State-Änderung manuell steuern)
        
        for i in range(len(obs_action)):
            obs = obs_action[i][:obs_length]
            action = obs_action[i][obs_length:obs_length + action_length]
            
            # Manuelle Setzung des States in die Umgebung (Wichtig!)
            env.state = obs  # Setze den aktuellen Zustand der Umgebung (bei CartPole ist `state` ein Array)
            
            # Schritte ausführen (mit gespeicherter Action)
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                break

        env.close()