import os 
import gym
import gym_kuka_mujoco
import numpy as np
import argparse
import json

from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy as AC_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy

from experiment_files import new_experiment_dir

def callback(_locals, _globals, log_dir):
    """
    Callback called at each gradient update.
    """
    # Get the current update step.
    n_update = _locals['update']

    # Save on the first update and every 10 updates after that.
    if (n_update == 1) or (n_update % 10 == 0):
        checkpoint_save_path = os.path.join(log_dir, 'model_checkpoint_{}.pkl'.format(n_update))
        _locals['self'].save(checkpoint_save_path)

def run_learn(params):
    '''
    Runs the learning experiment defined by the params dictionary.
    '''
    # Unpack options
    learning_options = params['learning_options']
    actor_options = params['actor_options']
    save_path = new_experiment_dir(params)
    os.makedirs(save_path, exist_ok=True)

    # Save the parameters that will generate the model
    params_save_path = os.path.join(save_path,'params.json')
    with open(params_save_path, 'w') as f:
        json.dump(params, f, sort_keys = True, indent = 4, ensure_ascii=False)

    # Generate vectorized environment.
    envs = [gym.make(params['env']) for _ in range(params['n_env'])]
    env = SubprocVecEnv([lambda: e for e in envs])

    # Create the actor and learn
    if params['alg'] == 'PPO2':
        model = PPO2(AC_MlpPolicy, env, tensorboard_log=save_path, **actor_options)
    elif params['alg'] == 'SAC':
        model = SAC(SAC_MlpPolicy, env, tensorboard_log=save_path, **actor_options)
    else:
        raise NotImplementedError
    
    learn_callback = lambda l, g: callback(l, g, save_path)
    model.learn(callback=learn_callback, **learning_options)

    # Save the model
    model_save_path = os.path.join(save_path,'model')
    model.save(model_save_path)

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
