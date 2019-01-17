import os
import argparse
import json
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

import gym
import gym_kuka_mujoco

from experiment_files import get_experiment_dirs, get_latest_checkpoint, get_params

def load_params(params_path):
    with open(params_path) as f:
        data = json.load(f)
    return data

def load_model(checkpoint_path, params):
    orig_env = gym.make(params['env'])
    env = DummyVecEnv([lambda:orig_env])

    if params['alg'] == 'PPO2':
        model = PPO2.load(checkpoint_path, env=env)
    else:
        raise NotImplementedError

    return orig_env, model
    replay_model(orig_env, model)

def replay_model(env, model):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, info = env.step(clipped_action, render=True)
        if done:
            env.reset()

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='The directory of the experiment.')

    args = parser.parse_args()

    # Load the model.
    experiment_dir = get_experiment_dirs(args.directory)[0]
    checkpoint_path = get_latest_checkpoint(experiment_dir)
    params_path = get_params(experiment_dir)

    params = load_params(params_path)
    env, model = load_model(checkpoint_path, params)

    # Replay model.
    replay_model(env, model)
