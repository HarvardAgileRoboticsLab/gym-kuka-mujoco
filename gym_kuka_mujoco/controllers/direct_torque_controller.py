import numpy as np
import gym

from gym_kuka_mujoco.utils.mujoco_utils import kuka_subtree_mass
from .base_controller import BaseController
from . import register_controller

class DirectTorqueController(BaseController):
    '''
    A simple controller that takes raw torque actions.
    '''
    def __init__(self, sim, action_scaling=10.):
        super(DirectTorqueController, self).__init__(sim)

        # Scale the actions proportionally to the subtree mass.
        true_subtree_mass = kuka_subtree_mass(sim.model)
        normalized_subtree_mass = true_subtree_mass / np.max(true_subtree_mass)
        self.action_scaling = action_scaling * normalized_subtree_mass

        # Scale the action space to the new scaling.
        low = sim.model.actuator_ctrlrange[:, 0]/action_scaling
        high = sim.model.actuator_ctrlrange[:, 1]/action_scaling
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_action(self, action):
        self.torque = action*self.action_scaling

    def get_torque(self):
        return self.torque

class SACTorqueController(DirectTorqueController):
    '''
    A simple controller that takes raw torque actions.
    '''
    def __init__(self, sim, action_limit=1., **kwargs):
        super(SACTorqueController, self).__init__(sim, **kwargs)

        # Reduce the torque limits.
        limited_low = self.action_space.low*action_limit
        limited_high = self.action_space.high*action_limit
        self.action_space = gym.spaces.Box(limited_low, limited_high, dtype=np.float32)

register_controller(DirectTorqueController, 'DirectTorqueController')
register_controller(SACTorqueController, 'SACTorqueController')