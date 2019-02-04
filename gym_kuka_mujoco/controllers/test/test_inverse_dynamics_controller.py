import numpy as np
from common import create_sim
from gym_kuka_mujoco.envs import KukaEnv
from gym_kuka_mujoco.controllers import InverseDynamicsController

def test_inverse_dynamics_controller():
    options = dict()
    options['kp_id'] = 100.
    options['kd_id'] = 'auto'

    sim = create_sim()
    controller = InverseDynamicsController(sim, **options)
    
    sim.data.qpos[:] = np.ones(7)
    sim.data.qvel[:] = np.ones(7)
    controller.set_action(np.zeros(7))

    controller.get_torque()



def test_relative_inverse_dynamics_controller():
    pass

if __name__ == "__main__":
    test_inverse_dynamics_controller()
    test_relative_inverse_dynamics_controller()