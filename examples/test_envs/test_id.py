import gym
from gym.spaces import Box
from stable_baselines import PPO2, SAC

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy as AC_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy
import matplotlib.pyplot as plt

import gym_kuka_mujoco
import numpy as np


def test_random_actions():
    '''
    Visualize random actions in the action space.
    '''
    env = gym.make('DiffIdControlledKukaMujoco-v0')
    print('action_space high: {}'.format(env.action_space.high))
    print('action_space low:  {}'.format(env.action_space.low))

    env.reset()
    try:
        while True:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action, render=True)
    finally:
        env.close()


def test_predict_PPO():
    '''
    Visualize predictions from a random policy.
    '''
    orig_env = gym.make('DiffIdControlledKukaMujoco-v0')
    env = DummyVecEnv([lambda: orig_env])
    model = PPO2(AC_MlpPolicy, env)
    obs = orig_env.reset()
    try:
        while True:
            action, _ = model.predict(obs)
            obs, rew, done, info = orig_env.step(action, render=True)
            if done:
                obs = orig_env.reset()
    finally:
        env.close()
        orig_env.close()

def test_predict_SAC():
    '''
    Visualize predictions from a random policy.
    '''
    orig_env = gym.make('DiffIdControlledKukaMujoco-v0')
    env = DummyVecEnv([lambda: orig_env])
    model = SAC(SAC_MlpPolicy, env)
    obs = orig_env.reset()
    try:
        while True:
            action, _ = model.predict(obs)
            print(action)
            obs, rew, done, info = orig_env.step(action, render=True)
            if done:
                obs = orig_env.reset()
    finally:
        env.close()
        orig_env.close()


if __name__ == "__main__":
    # test_random_actions()
    # test_predict_PPO()
    test_predict_SAC()
    # test_predict_distribution()
    # test_simple_controller()
    plt.show()
