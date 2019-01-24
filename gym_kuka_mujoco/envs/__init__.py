from gym_kuka_mujoco.envs.kuka_env import KukaEnv, KukaEnvSAC
from gym_kuka_mujoco.envs.id_controlled_kuka_env import (
    IdControlledKukaEnv, DiffIdControlledKukaEnv)
from gym_kuka_mujoco.envs.peg_insertion_env import PegInsertionEnv, RemoteCenterPegInsertionEnv
from gym_kuka_mujoco.envs.remote_center_controlled_kuka_env import RemoteCenterControlledKukaEnv#, RemoteCenterResidualControlledKukaEnv

# tests
from gym_kuka_mujoco.envs.peg_insertion_env import (
    QuadraticCostPegInsertionEnv,
    LinearCostPegInsertionEnv,
    QuadraticLogarithmicCostPegInsertionEnv,
    QuadraticSparseCostPegInsertionEnv,
    QuadraticRegularizedCostPegInsertionEnv
)