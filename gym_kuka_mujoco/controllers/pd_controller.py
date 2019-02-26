import numpy as np
import gym

from gym_kuka_mujoco.utils.mujoco_utils import kuka_subtree_mass, get_qpos_indices, get_qvel_indices, get_actuator_indices
from .base_controller import BaseController
from . import register_controller


def stable_critical_damping(kp, subtree_mass, dt):
    return np.minimum(kp * dt, 2 * np.sqrt(subtree_mass * kp))


class PDController(BaseController):
    '''
    A Proportional Derivative Controller.
    '''

    def __init__(self,
                 sim,
                 action_scale=1.,
                 action_limit=1.,
                 controlled_joints=None,
                 kp=3.,
                 set_velocity=False):
        super(PDController, self).__init__(sim)

        # Get the controlled joints
        if controlled_joints:
            self.controlled_joints = get_qpos_indices(sim.model, controlled_joints)
        else:
            self.controlled_joints = range(sim.model.nq)

        assert sim.model.nu == len(self.controlled_joints), \
            "The number of controlled joints ({}) should match the number of actuators in the model ({})".format(
            len(self.controlled_joints), sim.model.nu)

        # Scale the action space to the new scaling.
        self.set_velocity = set_velocity
        self.action_scale = action_scale
        low_pos = sim.model.jnt_range[self.controlled_joints, 0] / action_scale
        high_pos = sim.model.jnt_range[self.controlled_joints, 1] / action_scale

        low_pos[self.sim.model.jnt_limited[self.controlled_joints] == 0] = -np.inf
        high_pos[self.sim.model.jnt_limited[self.controlled_joints] == 0] = np.inf

        if self.set_velocity:
            low_vel = np.ones_like(low_pos) / action_scale
            high_vel = np.ones_like(low_pos) / action_scale

            low = np.concatenate([low_pos, low_vel])
            high = np.concatenate([high_pos, high_vel])
        else:
            low = low_pos
            high = high_pos

        # Scale the actions proportionally to the subtree mass.
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

        # set the proportional control constants, try to make the critically damped
        subtree_mass = kuka_subtree_mass(sim.model)
        dt = sim.model.opt.timestep
        self.kp = kp * subtree_mass
        self.kd = stable_critical_damping(kp, subtree_mass, dt)
        self.sim = sim

        # Initialize the setpoints
        self.qpos_setpoint = np.zeros(sim.model.nu)
        self.qvel_setpoint = np.zeros(sim.model.nu)

    def set_action(self, action):
        '''
        Sets the setpoints for the PD Controller.
        '''
        action = action * self.action_scale

        nu = len(self.controlled_joints)
        self.qpos_setpoint = action[0:nu]
        if self.set_velocity:
            self.qvel_setpoint = action[nu:2 * nu]

    def get_torque(self):
        '''
        Computes the torques from the setpoints and the current state.
        '''
        return self.kp * (
            self.qpos_setpoint - self.sim.data.qpos[self.controlled_joints]
        ) + self.kd * (
            self.qvel_setpoint - self.sim.data.qvel[self.controlled_joints])


class RelativePDController(PDController):
    def set_action(self, action):
        action = action * self.action_scale

        nu = len(self.controlled_joints)
        # Set the setpoint difference from the current position.
        self.qpos_setpoint = action[0:nu] + \
            self.sim.data.qpos[self.controlled_joints]
        if self.set_velocity:
            self.qvel_setpoint = action[nu:2 * nu]


register_controller(PDController, 'PDController')
register_controller(RelativePDController, 'RelativePDController')
