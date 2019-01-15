import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy
import matplotlib.pyplot as plt

import gym_kuka_mujoco
import numpy as np

def test_predict():
    env = gym.make('KukaMujoco-v0')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, training=True, norm_reward=False, clip_obs=np.inf, clip_reward=np.inf)
    model = PPO2(MlpPolicy, env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rew, done, info = env.step(action)
        env.render()

def test_simple_controller():
    env = gym.make('KukaMujoco-v0')
    env.time_limit = 25
    done = False
    obs = env.reset()
    positions = []
    rewards = []
    while not done:
        action = -obs[:7] -obs[7:14]
        print(obs)
        obs, rew, done, info = env.step(action, render=True)
        rewards.append(rew)
        positions.append(obs[:7])
        # if done:
            # env.reset()

    plt.figure()
    plt.plot(positions)
    plt.title('Simple Controller Trajectory')

def test_random_actions():
    env = gym.make('KukaMujoco-v0')
    print('action_space high: {}'.format(env.action_space.high))
    print('action_space low:  {}'.format(env.action_space.low))

    env.reset()
    while True:
        action = env.action_space.sample()/10
        obs, rew, done, info = env.step(action, render=True)
    
if __name__=="__main__":
    test_predict()
    # test_simple_controller()
    # test_random_actions()
    plt.show()

