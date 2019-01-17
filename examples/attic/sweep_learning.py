import os 
from datetime import datetime

import numpy as np

import gym
import gym_kuka_mujoco

from mpi4py import MPI

from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_mujoco_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

def make_mujoco_env(env_id, seed, allow_early_resets=True):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed + 10000 * rank)
    env = gym.make(env_id)
    env.seed(seed)
    return env

norm_obs = True
norm_reward = False

for lr in np.logspace(-4.5,-2.5,3):
    # env_name = 'CartPole-v1'
    env_name = 'PegInsertionBigHole-v0'
    num_env = 8
    # num_env = 4

    description = 'lr={},norm_obs={},norm_reward={}'.format(lr,norm_obs,norm_reward)

    print(description)

    date, time = datetime.now().isoformat().split('T')
    tensorboard_logdir = os.path.join(
        os.environ['OPENAI_LOGDIR'],
        date,
        time,
        'parameter_sweep')

    actor_options = {
        'learning_rate': lr,
        'gamma':1.,
        'verbose':0,
        'n_steps':100,
        'ent_coef':0.,
        'max_grad_norm':1e2,
    }

    description = ','.join(['{}={}'.format(k,v) for k, v in actor_options.items()])
    description += ',num_env={},norm_obs={},norm_reward={}'.format(num_env, norm_obs, norm_reward)

    learning_options = {
        'total_timesteps': int(1e6)
    }

    # Wrap in a try statement to close the environment properly in case of keyboard interrupt.
    try:
        envs = [make_mujoco_env(env_name, 2) for _ in range(num_env)]
        # env = DummyVecEnv([lambda: env for env in envs])
        env = SubprocVecEnv([lambda: env for env in envs])
        env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
        
        # Create the actor and learn
        actor_options['tensorboard_log'] = os.path.join(tensorboard_logdir,env_name)
        model = PPO2(MlpPolicy, env, **actor_options)
        # model = PPO2(MlpLstmPolicy, env, **actor_options)
        model.learn(**learning_options, tb_log_name=description)
    finally:
        env.close()