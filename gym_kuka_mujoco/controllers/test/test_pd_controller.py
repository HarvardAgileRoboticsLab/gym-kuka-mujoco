import numpy as np

from gym_kuka_mujoco.envs import KukaEnv
from gym_kuka_mujoco.controllers import PDController

from common import create_sim


def test_pd_controller():
    sim = create_sim()
    options = dict()
    options['set_velocity'] = True

    # Test the null action with velocity control enabled
    controller = PDController(sim, **options)
    assert len(controller.action_space.low) == 14
    
    null_action = np.zeros(14)
    controller.set_action(null_action)
    assert np.all(controller.get_torque() == np.zeros(7))

    # Test the null action with no velocity control
    options = dict()
    options['set_velocity'] = False

    controller = PDController(sim, **options)
    assert len(controller.action_space.low) == 7
    
    # The null action at the origin should be zero
    null_action = np.zeros(7)
    controller.set_action(null_action)
    assert np.all(controller.get_torque() == np.zeros(7))


    # # Test the action space.
    # # import pdb; pdb.set_trace()
    # expected_action_limits = 300.*np.ones(7)
    # assert np.allclose(controller.action_space.low, -expected_action_limits)
    # assert np.allclose(controller.action_space.high, expected_action_limits)

    # # Test that the torque is linear in the action.
    # action_1 = np.array([1., 2., 3., 4., 5., 6., 7.])
    # action_2 = 2*np.array(action_1)
    # controller.set_action(action_1)
    # torque_1 = controller.get_torque()
    # controller.set_action(action_2)
    # torque_2 = controller.get_torque()

    # controller.set_action(action_1 + action_2)
    # assert np.allclose(controller.get_torque(), torque_1 + torque_2)

    # # Test that the action scaling works.
    # options_2 = dict()
    # options_2["action_scaling"] = 2.
    # sim_2 = create_sim()
    # controller_2 = DirectTorqueController(sim_2, **options_2)

    # controller_2.set_action(action_1)
    # torque_ratio = controller_2.get_torque()/torque_1
    # scaling_ratio = options_2["action_scaling"]/options["action_scaling"]
    # assert np.allclose(torque_ratio, scaling_ratio)

if __name__ == '__main__':
    test_pd_controller()
