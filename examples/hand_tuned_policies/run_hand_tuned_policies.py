import argparse
import os
import commentjson
from stable_baselines import PPO2, SAC
from gym_kuka_mujoco.envs import *
from stable_baselines.common.vec_env import DummyVecEnv

# Add the parent folder to the python path for imports.
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from play_model import replay_model
from impedance_peg_insertion import ManualImpedancePegInsertionPolicy

if __name__ == '__main__':
    import warnings
    
    # Setup command line arguments.
    parser = argparse.ArgumentParser(description='Runs a learning example on a registered gym environment.')
    parser.add_argument('--param_file',
                        type=str,
                        help='the parameter file to use')
    args = parser.parse_args()

    # Load the learning parameters from a file.
    param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'param_files')
    param_file = os.path.join(param_dir, args.param_file)
    with open(param_file) as f:
        params = commentjson.load(f)

    if params['env'] == "PegInsertionEnv":
        if params['env_options']['controller'] == "ImpedanceControllerV2":
            model = ManualImpedancePegInsertionPolicy()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Visualize.
    env_cls = globals()[params['env']]
    env = env_cls(**params['env_options'])
    
    replay_model(env, model)