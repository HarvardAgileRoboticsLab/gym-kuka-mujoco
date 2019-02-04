import numpy as np
from common import create_sim
from gym_kuka_mujoco.envs import KukaEnv
from gym_kuka_mujoco.controllers import InverseDynamicsController, RelativeInverseDynamicsController

def test_inverse_dynamics_controller():
    options = dict()
    options['kp_id'] = 100.
    options['kd_id'] = 'auto'

    sim = create_sim()
    controller = InverseDynamicsController(sim, **options)
    
    sim.data.qpos[:] = np.zeros(7)
    sim.data.qvel[:] = np.ones(7)
    controller.set_action(np.zeros(7))

    base_torque = controller.get_torque()

    # Test that coriolis terms are being calculated.
    assert not np.any(np.isclose(base_torque, np.zeros(7)))

    # Test that the torque is linear in the error
    controller.set_action(np.ones(7))
    torque_1 = controller.get_torque()
    controller.set_action(2*np.ones(7))
    torque_2 = controller.get_torque()
    
    assert not np.allclose(torque_1, base_torque)
    assert np.allclose(2*(torque_1-base_torque), torque_2-base_torque)

    # Test that the torque is state dependent
    sim.data.qpos[:] = np.array([.1, .2, .3, .4, .5, .6, .7])
    controller.set_action(np.zeros(7))
    assert not np.allclose(controller.get_torque(), base_torque)

def test_relative_inverse_dynamics_controller():
    sim = create_sim()
    controller = RelativeInverseDynamicsController(sim)

    sim.data.qpos[:] = np.zeros(7)
    sim.data.qvel[:] = np.zeros(7)

    # Test that the torques are non-zero.
    setpoint = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    controller.set_action(setpoint)
    torque = controller.get_torque()

    assert not np.any(np.isclose(np.zeros(7), torque))

    # Test that the torques are zero when the state is at the setpoint.
    sim.data.qpos[:] = setpoint
    torque = controller.get_torque()
    assert np.allclose(np.zeros(7), torque)


if __name__ == "__main__":
    test_inverse_dynamics_controller()
    test_relative_inverse_dynamics_controller()