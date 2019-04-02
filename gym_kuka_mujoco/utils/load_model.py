# Native libraries.
import os

# Installed libraries.
import commentjson
import numpy as np
from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv

# Local imports.
from gym_kuka_mujoco.envs import *


def load_params(params_path):
    with open(params_path) as f:
        data = commentjson.load(f)
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