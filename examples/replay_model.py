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
        obs, reward, done, info = env.step(action, render=True)
        if done:
            env.reset()

if __name__=='__main__':
    # Visualize the solution
    model_path = os.path.join(os.environ['OPENAI_LOGDIR'],
                              'stable',
                              '2019-01-06',
                              '22:20:55.849317',
                              'cirriculum_learning',
                              'PegInsertionNoHole-v0',
                              'model.pkl')
    running_average_path = os.path.join(os.environ['OPENAI_LOGDIR'],
                          'stable',
                          '2019-01-06',
                          '22:20:55.849317',
                          'cirriculum_learning',
                          'PegInsertionNoHole-v0')

    orig_env = gym.make('PegInsertionNoHole-v0')
    env = DummyVecEnv([lambda: orig_env])
    env = VecNormalize(env, training=False, norm_reward=False, clip_obs=np.inf, clip_reward=np.inf)
    env.load_running_average(running_average_path)
    model = PPO2(MlpPolicy, env)
    model.load(model_path)
    replay_model(orig_env, model)