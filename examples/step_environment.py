import argparse
import os
import commentjson
import numpy as np
from stable_baselines import PPO2, SAC
from gym_kuka_mujoco.envs import *
from stable_baselines.common.vec_env import DummyVecEnv
from play_model import replay_model

if __name__ == '__main__':    
    # Setup command line arguments.
    parser = argparse.ArgumentParser(description='Runs a learning example on a registered gym environment.')
    parser.add_argument('--default_name',
                        type=str,
                        default='KukaMujoco-v0:PPO2',
                        help='the name of the default entry to use')
    parser.add_argument('--param_file',
                        type=str,
                        help='the parameter file to use')
    parser.add_argument('--deterministic',
                        action='store_true',
                        help='the randomness of the policy to visualize')
    args = parser.parse_args()

    # Load the learning parameters from a file.
    param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'param_files')
    if args.param_file is None:
        default_path = os.path.join(param_dir, 'default_params.json')
        with open(default_path) as f:
            params = commentjson.load(f)[args.default_name]
    else:
        param_file = os.path.join(param_dir, args.param_file)
        with open(param_file) as f:
            params = commentjson.load(f)

    # Visualize.
    env_cls = globals()[params['env']]
    env = env_cls(**params['env_options'])
    vec_env = DummyVecEnv([lambda: env])
    
    action = np.array([ 0.48196694,  1.832208  , -0.2773429 , -0.5335685 , -0.7700685 , -0.17673267,  1.0382922 ])

    # Repeat that action forever
    while True:
        env.reset()
        env.step(action, render=True)