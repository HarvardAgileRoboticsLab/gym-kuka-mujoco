import gym
from gym.spaces import Box
from stable_baselines import PPO2, SAC

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy as AC_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy
import matplotlib.pyplot as plt

import gym_kuka_mujoco
import numpy as np
from warnings import warn

'''
Environment diagnostics
'''

def validate_env(env_id):
    # Validate the action space.
    env = gym.make(env_id)
    high = env.action_space.high
    low = env.action_space.low
    assert  not np.all(np.isnan(high)), "{} has NaNs in the action space.".format(env_id)
    assert  not np.all(np.isnan(low)), "{} has NaNs in the action space.".format(env_id)
    assert np.all(env.action_space.high >= env.action_space.low), "{} has invalid action ranges.".format(env_id)

    if np.any(low > -1.0) or np.any(high < 1.0):
        warn("The action space for {} should contain at least (-1, 1), consider rescaling.")


def validate_PPO(env_id):
    policy_stats = test_random_policy_PPO(env_id)

    # Check clipping on the action space
    if np.any(policy_stats['invalid_actions']):
        warn("The policy is not respecting the action space limits: {}".format(env_id))

    if np.any(policy_stats['frac_clip_high'] > 0.05) or \
       np.any(policy_stats['frac_clip_low'] > 0.05):
        warn("The action space is clipping the policy: {env_id}\n"
             "\tfrac_clip_high:  {frac_clip_high}\n"
             "\tfrac_clip_low:   {frac_clip_low}".format(env_id=env_id, **policy_stats))

    # Check scaling of the actions and observations
    if np.any(policy_stats['act_std'] > 10.) or \
       np.any(policy_stats['act_std'] < .1) or \
       np.any(policy_stats['mean_ep_act_std'] > 10.) or \
       np.any(policy_stats['mean_ep_act_std'] < .1):
        warn("The random policy is giving poorly scaled actions: {env_id}\n"
             "\tact_mean:        {act_mean}\n"
             "\tact_std:         {act_std}\n"
             "\tmean_ep_act_std: {mean_ep_act_std}".format(env_id=env_id, **policy_stats))

    if np.any(policy_stats['obs_std'] > 10.) or \
       np.any(policy_stats['obs_std'] < .1) or \
       np.any(policy_stats['mean_ep_obs_std'] > 10.) or \
       np.any(policy_stats['mean_ep_obs_std'] < .1):
        warn("The random policy is recieving poorly scaled observations: {env_id}\n"
             "\tobs_mean:        {obs_mean}\n"
             "\tobs_std:         {obs_std}\n"
             "\tmean_ep_obs_std: {mean_ep_obs_std}".format(env_id=env_id, **policy_stats))

    # Check the reward
    if policy_stats['rew_std'] < 1e-5 or \
       policy_stats['mean_ep_rew_std'] < 1e-5:
        warn("The random policy is recieving near constant reward: {env_id}"
             "\trew_mean:        {rew_mean}\n"
             "\trew_std:         {rew_std}\n"
             "\tmean_ep_rew_std: {mean_ep_rew_std}".format(env_id=env_id, **policy_stats))

    return policy_stats

def get_statistics(env, actions, observations, rewards, dones, ep_ids):
    actions = np.array(actions)
    observations = np.array(observations)
    rewards = np.array(rewards)
    dones = np.array(dones)
    ep_ids = np.array(ep_ids)
    
    stats = {}
    stats['n_steps'] = len(actions)
    stats['n_episodes'] = np.sum(dones)
    stats['mean_ep_len'] = stats['n_steps']/stats['n_episodes']

    stats['act_mean'] = np.mean(actions, axis=0)
    stats['act_std'] = np.std(actions, axis=0)
    ep_act_stds = [np.std(actions[ep_ids==i], axis=0) for i in range(stats['n_episodes'])]
    stats['mean_ep_act_std'] = np.mean(ep_act_stds, axis=0)
   
    stats['obs_mean'] = np.mean(observations, axis=0)
    stats['obs_std'] = np.std(observations, axis=0)
    ep_obs_stds = [np.std(observations[ep_ids==i], axis=0) for i in range(stats['n_episodes'])]
    stats['mean_ep_obs_std'] = np.mean(ep_obs_stds, axis=0)

    stats['rew_mean'] = np.mean(rewards)
    stats['rew_std'] = np.std(rewards)
    stats['mean_ep_rew'] = np.sum(rewards)/stats['n_episodes']
    ep_rew_stds = [np.std(rewards[ep_ids==i], axis=0) for i in range(stats['n_episodes'])]
    stats['mean_ep_rew_std'] = np.mean(ep_rew_stds, axis=0)

    clip_high = env.action_space.high == actions
    clip_low = env.action_space.low == actions

    stats['frac_clip_high'] = np.mean(clip_high, axis=0)
    stats['frac_clip_low'] = np.mean(clip_low, axis=0)

    invalid_action = np.logical_or(env.action_space.high < actions, env.action_space.low > actions)
    stats['invalid_actions'] = np.mean(invalid_action, axis=0)

    return stats

def print_statistics(stats):
    print(
        "n_steps:         {n_steps}\n"
        "n_episodes:      {n_episodes}\n"
        "mean_ep_len:     {mean_ep_len}\n"
        "rew_mean:        {rew_mean}\n"
        "rew_std:         {rew_std}\n"
        "mean_ep_rew_std: {mean_ep_rew_std}\n"
        "act_mean:        {act_mean}\n"
        "act_std:         {act_std}\n"
        "mean_ep_act_std: {act_std}\n"
        "obs_mean:        {obs_mean}\n"
        "obs_std:         {obs_std}\n"
        "mean_ep_obs_std: {obs_std}".format(**stats))


'''
Environment data collection
'''

def test_random_actions(env_id, n_steps=1000):
    '''
    Collect data from trials taking random actions in the action space.
    '''
    env = gym.make(env_id)

    actions = []
    observations = []
    rewards = []
    dones = []
    ep_ids = []
    ep_count = 0


    for i in range(n_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        
        actions.append(action)
        rewards.append(rew)
        observations.append(obs)
        dones.append(done)
        ep_ids.append(ep_count)

        if done:
            ep_count += 1
            env.reset()

    stats = get_statistics(env, actions, observations, rewards, dones, ep_ids)
    env.close()
    
    return stats

def test_random_policy_PPO(env_id, n_steps=1000):
    '''
    Collect data from trials taking random actions in the action space.
    '''
    orig_env = gym.make(env_id)
    env = DummyVecEnv([lambda: orig_env])
    model = PPO2(AC_MlpPolicy, env)

    actions = []
    observations = []
    rewards = []
    dones = []
    ep_ids = []
    ep_count = 0

    obs = orig_env.reset()
    for i in range(n_steps):
        action, _ = model.predict(obs)
        obs, rew, done, info = orig_env.step(action)
        
        actions .append(action)
        rewards.append(rew)
        observations.append(obs)
        dones.append(done)
        ep_ids.append(ep_count)
        
        if done:
            ep_count +=1
            obs = orig_env.reset()

    stats = get_statistics(env, actions, observations, rewards, dones, ep_ids)
    env.close()
    orig_env.close()
    return stats

'''
Visualization
'''

def vis_random_actions(env_id):
    '''
    Visualize random actions in the action space.
    '''
    env = gym.make(env_id)

    env.reset()
    try:
        while True:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action, render=True)
    finally:
        env.close()

def vis_random_policy_PPO(env_id):
    '''
    Visualize predictions from a random policy.
    '''
    orig_env = gym.make(env_id)
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

def vis_random_policy_SAC(env_id):
    '''
    Visualize predictions from a random policy.
    '''
    orig_env = gym.make(env_id)
    env = DummyVecEnv([lambda: orig_env])
    model = SAC(SAC_MlpPolicy, env)
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


if __name__ == "__main__":
    PPO_envs = [
        # 'KukaMujoco-v0',
        # 'IdControlledKukaMujoco-v0',
        # 'DiffIdControlledKukaMujoco-v0',
        # 'PegInsertion-v0',
        # 'PegInsertionNoHole-v0',
        'RemoteCenterControlledKukaMujoco-v0',
        # 'RemoteCenterPegInsertion-v0'
    ]

    np.set_printoptions(linewidth=1000)
    for env_id in PPO_envs:
        validate_env(env_id)
        stats = validate_PPO(env_id)
        # print_statistics(stats)

