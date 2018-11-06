from gym.envs.registration import register

register(
    id='KukaMujoco-v0',
    entry_point='gym_kuka_mujoco.envs:KukaEnv',
)

register(
    id='IdControlledKukaMujoco-v0',
    entry_point='gym_kuka_mujoco.envs:IdControlledKukaEnv',
)

register(
    id='DiffIdControlledKukaMujoco-v0',
    entry_point='gym_kuka_mujoco.envs:DiffIdControlledKukaEnv',
)
