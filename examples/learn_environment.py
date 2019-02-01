import os 
import gym
import gym_kuka_mujoco
import numpy as np
import argparse
import commentjson

from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy as AC_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results

from experiment_files import new_experiment_dir
from play_model import replay_model

# from gym_kuka_mujoco.wrappers import TBWrapper
from gym_kuka_mujoco.wrappers import TBVecEnvWrapper

def PPO_callback(_locals, _globals, log_dir):
    """
    Callback called at each gradient update.
    """
    # Get the current update step.
    n_update = _locals['update']


    # Save on the first update and every 10 updates after that.
    if (n_update == 1) or (n_update % 10 == 0):
        checkpoint_save_path = os.path.join(log_dir, 'model_checkpoint_{}.pkl'.format(n_update))
        _locals['self'].save(checkpoint_save_path)


    # import pdb; pdb.set_trace()
    # load_results(log_dir)
    # if len(_locals.get('ep_infos',[])) > 0:
    #     if 'tip_distance' in _locals['ep_infos'][0]
    #         distances = np.array([info['tip_distance'] for i, info in _locals['ep_infos']])
    #         final_distances = distances[_locals['masks']]



def SAC_callback(_locals, _globals, log_dir):
    """
    Callback called at each gradient update.
    """
    # Get the current update step.
    # print('in callback')
    new_update = SAC_callback.n_updates < _locals['n_updates']
    if new_update:
        SAC_callback.n_updates = _locals['n_updates']

    # Save on the first update and every 10 updates after that.
    if new_update and ((SAC_callback.n_updates == 1) or (SAC_callback.n_updates % 1000 == 0)):
        print('new_update: {}'.format(SAC_callback.n_updates))
        checkpoint_save_path = os.path.join(log_dir, 'model_checkpoint_{}.pkl'.format(SAC_callback.n_updates))
        _locals['self'].save(checkpoint_save_path)
SAC_callback.n_updates = 0

def make_env(env_id, rank, save_path, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, save_path, info_keywords=env.info_keywords)
        return env
    set_global_seeds(seed)
    return _init

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
        commentjson.dump(params, f, sort_keys = True, indent = 4, ensure_ascii = False)

    # Generate vectorized environment.
    envs = [make_env(params['env'], i, save_path) for i in range(params['n_env'])]

    if params.get('vectorized', True):
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)
    env = TBVecEnvWrapper(env, save_path, info_keywords = params.get('info_keywords', tuple()))

    # Create the actor and learn
    if params['alg'] == 'PPO2':
        model = PPO2(params['policy_type'], env, tensorboard_log = save_path, **actor_options)
        learn_callback = lambda l, g: PPO_callback(l, g, save_path)
    elif params['alg'] == 'SAC':
        model = SAC(params['policy_type'], env, tensorboard_log = save_path, **actor_options)
        learn_callback = lambda l, g: SAC_callback(l, g, save_path)
    else:
        raise NotImplementedError
    
    print("Learning and recording to:\n{}".format(save_path))
    model.learn(callback = learn_callback, **learning_options)

    # Save the model
    model_save_path = os.path.join(save_path, 'model')
    model.save(model_save_path)

    return model

if __name__ == '__main__':
    import warnings
    
    # Setup command line arguments.
    parser = argparse.ArgumentParser(description='Runs a learning example on a registered gym environment.')
    parser.add_argument('--default_name',
                        type=str,
                        default='KukaMujoco-v0:PPO2',
                        help='the name of the default entry to use')
    parser.add_argument('--param_file',
                        type=str,
                        help='the parameter file to use')
    parser.add_argument('--filter_warning',
                        choices=['error','ignore','always','default','module','once'],
                        default='default',
                        help='the treatment of warnings')
    parser.add_argument('--debug',
                        action='store_true',
                        help='enables useful debug settings')
    args = parser.parse_args()

    # Change the warning behavior for debugging.
    warnings.simplefilter(args.filter_warning, RuntimeWarning)

    # Load the learning parameters from a file.
    if args.param_file is None:
        default_path = os.path.join('param_files', 'default_params.json')
        with open(default_path) as f:
            params = commentjson.load(f)[args.default_name]
    else:
        param_file = os.path.join('param_files', args.param_file)
        with open(param_file) as f:
            params = commentjson.load(f)
    
    # Override some arguments in debug mode
    if args.debug:
        params['vectorized'] = False

    # Learn.
    model = run_learn(params)
    
    # Visualize. 
    env = gym.make(params['env'])
    replay_model(env, model)
