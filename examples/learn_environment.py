import os 
from datetime import datetime
import gym
import gym_kuka_mujoco
import numpy as np
import argparse
import json

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy

def run_learn(params):
    '''
    Runs the learning experiment defined by the params dictionary.
    '''

    # Unpack options
    learning_options = params['learning_options']
    actor_options = params['actor_options']
    save_path = save_path_from_params(params)

    # Generate vectorized environment.
    envs = [gym.make(params['env']) for _ in range(params['n_env'])]
    env = SubprocVecEnv([lambda: e for e in envs])

    # Create the actor and learn
    model = PPO2(MlpPolicy, env, tensorboard_log=save_path, **actor_options)    
    model.learn(**learning_options)

    # Save the model
    model_save_path = os.path.join(save_path,'model')
    model.save(model_save_path)

    # Save the parameters that generated the model
    params_save_path = os.path.join(save_path,'params.json')
    with open(params_save_path, 'w') as f:
        json.dump(params, f, sort_keys = True, indent = 4, ensure_ascii=False)

    return model

def visualize_solution(params, model):
    '''
    Visualize the solution specified by the model.
    '''

    # Create the environment.
    env = gym.make(params['env'])
    env = DummyVecEnv([lambda: env])

    # Simulate forward.
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, rewards, dones, info = env.step(clipped_action)
        env.render()
        if dones[0]:
            env.reset()

def save_path_from_params(params):
    '''
    Generates the path to save the model and the experiment data from a
    dictionary of parameters.
    '''

    # Create a unique path based on the date and time of the experiment.
    day, time = datetime.now().isoformat().split('T')

    # Create a unique path based on a description of the experiment.
    description = ['alg={}'.format(params['alg']), 'env={}'.format(params['env'])]
    for k,v in params['learning_options'].items():
        description.append('{}={}'.format(k,v))

    for k,v in params['actor_options'].items():
        description.append('{}={}'.format(k,v))
    description = ','.join(description)
    
    # Contruct the path and return.
    save_path = os.path.join(os.environ['OPENAI_LOGDIR'], 'stable', day, time, description)    
    return save_path


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
    args = parser.parse_args()

    # Load the learning parameters from a file.
    if args.param_file is None:
        default_path = os.path.join('param_files', 'default_params.json')
        with open(default_path) as f:
            params = json.load(f)[args.default_name]
    else:
        param_file = os.path.join('param_files', args.param_file)
        with open(param_file) as f:
            params = json.load(f)
    
    # Learn and visualize.
    model = run_learn(params)
    visualize_solution(params, model)
