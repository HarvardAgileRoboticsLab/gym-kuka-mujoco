import os
from datetime import datetime
import gym
import gym_kuka_mujoco
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy

# Visualize the solution
env = gym.make('RemoteCenterControlledKukaMujoco-v0')
obs = env.reset()
env.time_limit = 20
while True:
    action = np.zeros(6)
    action = env.action_space.sample().astype(np.float64)
    action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, done, info = env.step(action, render=True)
    if done:
        env.reset()