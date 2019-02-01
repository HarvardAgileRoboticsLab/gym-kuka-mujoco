import gym
from .base_controller import BaseController

class DirectTorqueController(BaseController):
    '''
    A simple controller that takes raw torque actions.
    '''
    def __init__(self, action_space, torque_scaling=1.):
        super(DirectTorqueController, self).__init__()
        self.action_space = action_space
        self.torque_scaling = torque_scaling

    def set_action(self, action, sim):
        self.torque = action*self.torque_scaling

    def get_torque(self, sim):
        return self.torque