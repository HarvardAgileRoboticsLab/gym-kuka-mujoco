import gym
import gym_kuka_mujoco
import matplotlib.pyplot as plt

# Visualize the solution
env = gym.make('PegInsertionNoHole-v0')
obs = env.reset()

rewards = []
observations = []
cumulative_info = {}

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action, render=False)
    for key, val in info.items():
        cumulative_info[key] = cumulative_info.get(key,[]) + [val]
    observations.append(obs)
    rewards .append(reward)
    if done:
        env.reset()

print(cumulative_info)
plt.figure()
plt.plot(rewards, label='total')
for key, val in cumulative_info.items():
    plt.plot(val, label=key)
plt.legend()

position_observations = [obs[:7] for obs in observations]
velocity_observations = [obs[7:14] for obs in observations]
force_torque_observations = [obs[14:20] for obs in observations]

plt.figure()
plt.title('Position Observations')
plt.plot(position_observations)

plt.figure()
plt.title('Velocity Observations')
plt.plot(velocity_observations)

plt.figure()
plt.title('Force Torque Observations')
plt.plot(force_torque_observations)

plt.show()