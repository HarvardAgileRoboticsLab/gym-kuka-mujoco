import json
import os

default_params = {
    'KukaMujoco-v0:PPO2': {
        'env': 'KukaMujoco-v0',
        'alg': 'PPO2',
        'actor_options': {
            'learning_rate': 1e-3,
            'gamma': 1.,
            'n_steps': 2048,
            'ent_coef': 1e-2,
            'verbose': 0,
        },
        'learning_options': {
            'total_timesteps': int(2e7)
        },
        'n_env': 8,
    },
}

default_path = os.path.join('param_files', 'default_params.json')

if __name__ == '__main__':
    with open(default_path, 'w') as f:
        json.dump(
            default_params, f, sort_keys=True, indent=4, ensure_ascii=False)