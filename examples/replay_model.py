import os 
from datetime import datetime
import gym
import gym_kuka_mujoco
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy

def replay_model(env, model):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, info = env.step(clipped_action, render=True)
        if done:
            env.reset()

if __name__=='__main__':
    # Visualize the solution
    environment_name = 'PegInsertionNoHole-v0'
    environment_name = 'RemoteCenterControlledKukaMujoco-v0'
    environment_name = 'KukaMujoco-v0'
    running_average_path = os.path.join(os.environ['OPENAI_LOGDIR'],
                          'stable',
                          '2019-01-14',
                          # '17:48:38.178209',
                          '16:49:36.217527',
                          'cirriculum_learning',
                          environment_name)

    model_path = os.path.join(running_average_path,
                              'model.pkl')

    orig_env = gym.make(environment_name)
    env = DummyVecEnv([lambda: orig_env])
    # env = VecNormalize(env, training=False, norm_reward=False, clip_obs=np.inf, clip_reward=np.inf)
    # env.load_running_average(running_average_path)
    model = PPO2.load(model_path, env=env)
    replay_model(orig_env, model)