import numpy as np
import mujoco_py
from gym_kuka_mujoco.envs import KukaEnv
from gym_kuka_mujoco.controllers import DirectTorqueController

def create_env(controller_options):
    env = KukaEnv('DirectTorqueController', controller_options)
    return env

def test_torque_controller():
    options = dict()
    options["action_scaling"] = 1.
    env = create_env(options)

    controller = env.controller

    # Test the action space.
    # import pdb; pdb.set_trace()
    expected_action_limits = 300.*np.ones(7)
    assert np.allclose(controller.action_space.low, -expected_action_limits)
    assert np.allclose(controller.action_space.high, expected_action_limits)

    # Test that the torque is linear in the action.
    action_1 = np.array([1., 2., 3., 4., 5., 6., 7.])
    action_2 = 2*np.array(action_1)
    controller.set_action(action_1)
    torque_1 = controller.get_torque()
    controller.set_action(action_2)
    torque_2 = controller.get_torque()

    controller.set_action(action_1 + action_2)
    assert np.allclose(controller.get_torque(), torque_1 + torque_2)

    # Test that the action scaling works.
    options_2 = dict()
    options_2["action_scaling"] = 2.
    env_2 = create_env(options_2)
    controller_2 = env_2.controller

    controller_2.set_action(action_1)
    torque_ratio = controller_2.get_torque()/torque_1
    scaling_ratio = options_2["action_scaling"]/options["action_scaling"]
    assert np.allclose(torque_ratio, scaling_ratio)

if __name__ == '__main__':
    test_torque_controller()
