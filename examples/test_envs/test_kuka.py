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
    env = gym.make('KukaMujoco-v0')
    print('action_space high: {}'.format(env.action_space.high))
    print('action_space low:  {}'.format(env.action_space.low))

    env.reset()
    while True:
        action = env.action_space.sample()/10
        obs, rew, done, info = env.step(action, render=True)


def test_predict_PPO():
    '''
    Visualize predictions from a random policy.
    '''
    env = gym.make('KukaMujoco-v0')
    env = DummyVecEnv([lambda: env])
    model = PPO2(AC_MlpPolicy, env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rew, done, info = env.step(action)
        env.render()


def test_predict_SAC():
    '''
    Visualize predictions from a random policy.
    '''
    env = gym.make('KukaMujocoSAC-v0')
    model = SAC(SAC_MlpPolicy, env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rew, done, info = env.step(action, render=True)


def test_predict_distribution():
    '''
    Test that the distribution of outputs from a random policy is
    roughly a unit Gaussian.
    '''
    env = gym.make('KukaMujoco-v0')
    env = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, env)

    # Reserve space for the mean and covariance estimate.
    n_actions = env.action_space.low.size
    mean = np.zeros(n_actions)
    cov = np.zeros((n_actions, n_actions))

    N = 1000
    for i in range(N):
        obs = env.reset()
        action, _ = model.predict(obs)
        mean += action[0]
        cov += np.einsum('i,j->ij', action[0], action[0])

    mean /= N
    cov /= N
    cov = cov - np.einsum('i,j->ij', mean, mean)

    print('Mean: {}'.format(mean))
    print('Cov:\n{}'.format(cov))
    print('Sparse Cov:\n{}'.format((cov > 0.1).astype(np.float64)))


def test_simple_controller():
    '''
    Test that we can write a simple controller that can solve the task.
    '''
    env = gym.make('KukaMujoco-v0')
    env.time_limit = 10
    done = False
    obs = env.reset()
    positions = []
    rewards = []
    actions = []
    while not done:
        action = -obs[:7] -.2*obs[7:14]
        print(obs)
        obs, rew, done, info = env.step(action, render=True)
        rewards.append(rew)
        positions.append(obs[:7])
        actions.append(action)

    plt.figure()
    plt.subplot(311)
    plt.plot(positions)
    plt.subplot(312)
    plt.plot(rewards)
    plt.subplot(313)
    plt.plot(actions)
    plt.title('Simple Controller Trajectory')


if __name__ == "__main__":
    # test_random_actions()
    test_predict_SAC()
    # test_predict_distribution()
    # test_simple_controller()
    plt.show()
