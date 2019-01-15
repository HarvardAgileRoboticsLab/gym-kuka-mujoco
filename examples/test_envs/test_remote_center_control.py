import gym
import gym_kuka_mujoco
import numpy as np

def test_goal():
    env = gym.make('RemoteCenterControlledKukaMujoco-v0')
    env.set_state(np.zeros(7), np.zeros(7))
    obs, rew, done, info = env.step(np.zeros(6))
    assert np.allclose(env.sim.data.qpos, np.zeros(7))
    assert np.allclose(env.sim.data.qvel, np.zeros(7))
    assert np.allclose(rew, 0)

def test_random_fixed_point():
    qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    qvel = np.zeros(7)

    env = gym.make('RemoteCenterControlledKukaMujoco-v0')
    env.set_state(qpos, qvel)
    obs, rew, done, info = env.step(np.zeros(6))
    assert np.allclose(env.sim.data.qpos, qpos)
    assert np.allclose(env.sim.data.qvel, qvel)

if __name__=="__main__":
    test_goal()