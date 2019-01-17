import os 
from datetime import datetime
import gym
import gym_kuka_mujoco

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy

env_names = ['PegInsertion{}-v0'.format(n) for n in ['BigHole','MidHole','SmallHole','TinyHole','']]
num_env = 8

tensorboard_logdir = os.path.join(
    os.environ['OPENAI_LOGDIR'],
    'stable',
    datetime.now().isoformat(),
    'cirriculum_learning')

actor_options = {
    'learning_rate': lambda a: (1-a)*1e-4 + a*1e-3,
    'gamma':1,
    'verbose':0,
    'n_steps':256,
    'ent_coef':0,
}

learning_options = {
    'total_timesteps': int(1e5)
}

for i in range(1):
    # Create the environment
    print('Training on environment {}:'.format(env_names[i]))
    print('\t--setting up environments')
    envs = [gym.make(env_names[i]) for _ in range(num_env)]

    # Wrap in a try statement to close the environment properly.
    try:
        print('\t--instantiating vectorized environment')
        env = SubprocVecEnv([lambda: e for e in envs])
        env = VecNormalize(env, norm_reward=False)
        
        # Create the actor and learn
        print('\t--setting up actor')
        actor_options['tensorboard_log'] = os.path.join(tensorboard_logdir,env_names[i])
        print('\t--learning')
        if i==0:
            model = PPO2(MlpPolicy, env, **actor_options)
        else:
            model = PPO2.load(save_path, env=env, **actor_options)

        model.learn(**learning_options)
        save_path = os.path.join(actor_options['tensorboard_log'],'model')
        model.save(save_path)
    finally:
        env.close()


# Visualize the solution
env = gym.make(env_names[0])
env = DummyVecEnv([lambda: env])
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones[0]:
        env.reset()