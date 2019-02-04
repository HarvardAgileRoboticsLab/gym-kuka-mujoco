import numpy as np

from gym_kuka_mujoco.envs import KukaEnv
from gym_kuka_mujoco.controllers import DirectTorqueController, SACTorqueController

from common import create_sim


def test_direct_torque_controller():
    sim = create_sim()
    options = dict()
    options["action_scaling"] = 1.

    controller = DirectTorqueController(sim, **options)

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
    sim_2 = create_sim()
    controller_2 = DirectTorqueController(sim_2, **options_2)

    controller_2.set_action(action_1)
    torque_ratio = controller_2.get_torque()/torque_1
    scaling_ratio = options_2["action_scaling"]/options["action_scaling"]
    assert np.allclose(torque_ratio, scaling_ratio)

def test_sac_torque_controller():
    options = dict()
    options["limit_scale"] = 10.

    sim = create_sim()
    direct_controller = DirectTorqueController(sim) 
    sac_controller = SACTorqueController(sim, **options) 

    assert np.allclose(direct_controller.action_space.low, sac_controller.action_space.low*options["limit_scale"])
    assert np.allclose(direct_controller.action_space.high, sac_controller.action_space.high*options["limit_scale"])


if __name__ == '__main__':
    test_direct_torque_controller()
    test_sac_torque_controller()
