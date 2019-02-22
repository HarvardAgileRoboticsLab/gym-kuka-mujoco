import argparse
import os
import commentjson
import numpy as np
from stable_baselines import PPO2, SAC
from gym_kuka_mujoco.envs import *
from stable_baselines.common.vec_env import DummyVecEnv
from play_model import replay_model
import warnings
import tensorflow as tf

# suppress tensorflow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# def replay_model(env, model, deterministic=True, num_sims=2):
#     obs = env.reset()
#     distance = []
#     sims = 0
#     while True:
#         action, _states = model.predict(obs, deterministic=deterministic)
#         clipped_action = np.clip(action, env.action_space.low,
#                                  env.action_space.high)
#         obs, reward, done, info = env.step(clipped_action, render=False)
#         distances.append(info['tip_distance'])
#         if done:
#             obs = env.reset()
#             sims += 1
#         if sims >= num_sims:
#             break
#     return np.mean(distances)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
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
    parser.add_argument('--info_keywords',
                        type=str,
                        default='tip_distance',
                        help='a list of info keywords to collect statistics')
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


    # Collect the info keywords.
    if len(args.info_keywords):
        info_keywords = args.info_keywords.split(',')
    else:
        info_keywords = []

    # Report the data over a number of random initializations.
    iters = 5
    for i in range(iters):
        print('Iteration: {}'.format(i))

        # Create a random environment.
        if params['alg'] == 'PPO2':
            model = PPO2(params['policy_type'], vec_env, **params['actor_options'])
        elif params['alg'] == 'SAC':
            model = SAC(params['policy_type'], vec_env, **params['actor_options'])
        else:
            raise NotImplementedError
        
        # Collect data.
        infos = replay_model(env, model, deterministic=False, record=True, render=False, num_episodes=100)
        summary = dict()
        for key in info_keywords:
            data = [d[key] for d in infos]
            summary[key + "_mean"] = np.mean(data)
            summary[key + "_std"] = np.std(data)
            print("{}_mean: {}".format(key, np.mean(data)))
            print("{}_std: {}".format(key, np.std(data)))
