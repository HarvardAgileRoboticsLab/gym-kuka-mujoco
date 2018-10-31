from gym.envs.registration import register

register(
    id='KukaMujoco-v0',
    entry_point='gym_kuka_mujoco.envs:KukaEnv',
)