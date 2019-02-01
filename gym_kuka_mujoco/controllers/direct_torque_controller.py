import numpy as np
import gym
from .base_controller import BaseController

class DirectTorqueController(BaseController):
    '''
    A simple controller that takes raw torque actions.
    '''
    def __init__(self, env, action_scaling=10.):
        super(DirectTorqueController, self).__init__(env)

        # Scale the actions proportionally to the subtree mass.
        normalized_subtree_mass = env.subtree_mass() / np.max(env.subtree_mass())
        self.action_scaling = action_scaling * normalized_subtree_mass

        # Scale the action space to the new scaling.
        low = env.model.actuator_ctrlrange[:, 0]/action_scaling
        high = env.model.actuator_ctrlrange[:, 1]/action_scaling
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_action(self, action):
        self.torque = action*self.action_scaling

    def get_torque(self):
        return self.torque

class SACTorqueController(BaseController):
    '''
    A simple controller that takes raw torque actions.
    '''
    def __init__(self, env, action_scaling=10., limit_scale=30.):
        super(SACTorqueController, self).__init__(env)

        # Scale the actions proportionally to the subtree mass.
        normalized_subtree_mass = env.subtree_mass() / np.max(env.subtree_mass())
        self.action_scaling = action_scaling * normalized_subtree_mass

        # Reduce the torque limits.
        limited_low = env.model.actuator_ctrlrange[:, 0]/limit_scale
        limited_high = env.model.actuator_ctrlrange[:, 1]/limit_scale

        # Scale the action space to the new scaling.
        low = limited_low/action_scaling
        high = limited_high/action_scaling
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_action(self, action):
        self.torque = action*self.action_scaling

    def get_torque(self):
        return self.torque
