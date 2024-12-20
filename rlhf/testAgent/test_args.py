class TestArgs:
    num_envs = 2
    num_steps = 4
    segment_size = 2
    gamma = 0.99
    gae_lambda = 0.95
    seed = 42
    learning_rate = 0.001
    total_timesteps = 1000
    num_minibatches = 4
    torch_deterministic = True
    clip_coef = 0.2
    env_id = "HalfCheetah-v4"
    exp_name = "TestRun"
    capture_video = True
    cuda = False
