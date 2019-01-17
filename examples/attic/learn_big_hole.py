import os 
from datetime import datetime
import gym
import gym_kuka_mujoco
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy

num_env = 1

day, time = datetime.now().isoformat().split('T')
tensorboard_logdir = os.path.join(
    os.environ['OPENAI_LOGDIR'],
    'stable',
    day,
    time,
    'cirriculum_learning')

actor_options = {
    'learning_rate': 1e-3,
    # 'learning_rate': lambda a: a*1e-3 + (1-a)*1e-4,
    'gamma':1,
    'verbose':0,
    #'n_steps':256,
    'n_steps':2048,
    'ent_coef':0,
}

learning_options = {
    'total_timesteps': int(1e6)
}

# Create the environment
print('\t--setting up environments')
envs = [gym.make('PegInsertionHole0-v0') for _ in range(num_env)]

# Wrap in a try statement to close the environment properly.
try:
    print('\t--instantiating vectorized environment')
    env = SubprocVecEnv([lambda: e for e in envs])
    # env = DummyVecEnv([lambda: e for e in envs])
    env = VecNormalize(env, norm_reward=False, clip_obs=np.inf)

    # Create the actor and learn
    print('\t--setting up actor')
    actor_options['tensorboard_log'] = os.path.join(tensorboard_logdir,'PegInsertionHole0-v0')
    print('\t--learning')
    model = PPO2(MlpPolicy, env, **actor_options)
    
    model.learn(**learning_options)
    save_path = os.path.join(actor_options['tensorboard_log'],'model')
    model.save(save_path)
finally:
    env.close()


# Visualize the solution
env = gym.make('PegInsertionHole0-v0')
env = DummyVecEnv([lambda: env])
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones[0]:
        env.reset()