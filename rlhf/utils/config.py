sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'metrics/episodic_return', 'goal': 'maximize'},
    'parameters': {
        'dropout': {'values': [0.2, 0.3, 0.35]},
        'batch_size': {'values': [64, 128]},
        'segment_size': {'values': [50, 60]},
        'k': {'values': [3, 4, 5]},
        'gamma': {'values': [0.95, 0.97, 0.99, 1.0]},
        'clip_coef': {'values': [0.1, 0.2, 0.3]},
        'gae_lambda': {'values': [0.9, 0.95, 1.0]},
        'l2_regularization': {'min': 1e-6, 'max': 1e-3},
        'learning_rate': {'min': 3e-6, 'max': 3e-2},
        'anteil': {'values': [0.005, 0.01, 0.015, 0.02]},
    }
}