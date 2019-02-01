import os
import argparse
import json
import numpy as np

from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv

from gym_kuka_mujoco.envs import *

from experiment_files import (get_experiment_dirs, get_model,
                              get_latest_checkpoint, get_params)


def load_params(params_path):
    with open(params_path) as f:
        data = json.load(f)
    return data


def load_model(model_path, params):
    env_cls = globals()[params['env']]
    orig_env = env_cls(**params['env_options'])
    env = DummyVecEnv([lambda: orig_env])

    if params['alg'] == 'PPO2':
        model = PPO2.load(model_path, env=env)
    elif params['alg'] == 'SAC':
        model = SAC.load(model_path, env=env)
    else:
        raise NotImplementedError

    return orig_env, model
    replay_model(orig_env, model)


def replay_model(env, model, deterministic=True):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=deterministic)
        clipped_action = np.clip(action, env.action_space.low,
                                 env.action_space.high)
        obs, reward, done, info = env.step(clipped_action, render=True)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory', type=str, help='The directory of the experiment.')
    parser.add_argument(
        '--deterministic', action='store_true', help='Optionally simulate the deterministic system.')

    args = parser.parse_args()

    # Load the model if it's availeble, otherwise that latest checkpoint.
    experiment_dir = get_experiment_dirs(args.directory)[0]
    params_path = get_params(experiment_dir)
    params = load_params(params_path)

    model_path = get_model(experiment_dir)
    if model_path is None:
        model_path = get_latest_checkpoint(experiment_dir)

    env, model = load_model(model_path, params)

    # Replay model.
    replay_model(env, model, deterministic=args.deterministic)
