'''
A script that runs curriculum learning examples from a parameter file.
'''

# System imports.
import os
import argparse

# Installed imports.
import commentjson

# gym_kuka_mujoco imports.
from gym_kuka_mujoco.envs import *

# Local imports.
from play_model import replay_model
from learn_environment import run_learn


if __name__ == '__main__':
    import warnings
    
    # Setup command line arguments.
    parser = argparse.ArgumentParser(description='Runs a learning example on a registered gym environment.')
    parser.add_argument('param_file',
                        type=str,
                        help='the parameter file to use')
    args = parser.parse_args()

    # Load the learning parameters from a file.
    param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'param_files')
    param_file = os.path.join(param_dir, args.param_file)
    with open(param_file) as f:
            params_list = commentjson.load(f)

    # Learn.
    model = None
    for i, params in enumerate(params_list):
        # import pdb; pdb.set_trace()
        model = run_learn(params, model=model, run_count=i)
    
    # Visualize.
    params = params_list[-1]
    env_cls = globals()[params['env']]
    env = env_cls(**params['env_options'])
    replay_model(env, model)
