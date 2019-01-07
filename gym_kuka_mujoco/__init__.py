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

register(
    id='PegInsertion-v0',
    entry_point='gym_kuka_mujoco.envs:PegInsertionEnv',
)

for i in range(100):
    register(
        id='PegInsertionHole{}-v0'.format(i),
        entry_point='gym_kuka_mujoco.envs:PegInsertionEnv',
        kwargs={'hole_id' : i}
    )

register(
    id='PegInsertionNoHole-v0',
    entry_point='gym_kuka_mujoco.envs:PegInsertionEnv',
    kwargs={'hole_id' : -1}
)

# Remote Center Control

register(
    id='RemoteCenterControlledKukaMujoco-v0',
    entry_point='gym_kuka_mujoco.envs:RemoteCenterControlledKukaEnv'
)


register(
    id='RemoteCenterPegInsertion-v0',
    entry_point='gym_kuka_mujoco.envs:RemoteCenterPegInsertionEnv',
)

for i in range(100):
    register(
        id='RemoteCenterPegInsertionHole{}-v0'.format(i),
        entry_point='gym_kuka_mujoco.envs:RemoteCenterPegInsertionEnv',
        kwargs={'hole_id' : i}
    )

register(
    id='RemoteCenterPegInsertionNoHole-v0',
    entry_point='gym_kuka_mujoco.envs:RemoteCenterPegInsertionEnv',
    kwargs={'hole_id' : -1}
)

# register(
#     id='PegInsertionHugeHole-v0',
#     entry_point='gym_kuka_mujoco.envs:PegInsertionHugeHoleEnv',
# )

# register(
#     id='PegInsertionBigHole-v0',
#     entry_point='gym_kuka_mujoco.envs:PegInsertionBigHoleEnv',
# )

# register(
#     id='PegInsertionMidHole-v0',
#     entry_point='gym_kuka_mujoco.envs:PegInsertionMidHoleEnv',
# )

# register(
#     id='PegInsertionSmallHole-v0',
#     entry_point='gym_kuka_mujoco.envs:PegInsertionSmallHoleEnv',
# )

# register(
#     id='PegInsertionTinyHole-v0',
#     entry_point='gym_kuka_mujoco.envs:PegInsertionTinyHoleEnv',
# )
