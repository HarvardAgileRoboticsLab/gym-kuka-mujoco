import os

import numpy as np
import gym
import mujoco_py

from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from gym_kuka_mujoco.utils.mujoco_utils import kuka_subtree_mass, get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices
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
                 set_velocity=False,
                 gravity_comp_model_path=None):
        super(PDController, self).__init__(sim)

        self.gravity_comp = False
        if gravity_comp_model_path is not None:
            self.gravity_comp = True
            model_path = os.path.join(kuka_asset_dir(), gravity_comp_model_path)
            self.model = mujoco_py.load_model_from_path(model_path)
            self.gravity_comp_sim = mujoco_py.MjSim(self.model)
            assert self.model.nv == self.sim.model.nv, \
                "the model for control and simulation must have the same number of DOF"

        # Get the position, velocity, and actuator indices for the model.
        if controlled_joints is not None:
            self.sim_qpos_idx = get_qpos_indices(sim.model, controlled_joints)
            self.sim_qvel_idx = get_qvel_indices(sim.model, controlled_joints)
            self.sim_actuators_idx = get_actuator_indices(sim.model, controlled_joints)
            self.sim_joint_idx = get_joint_indices(sim.model, controlled_joints)
            if hasattr(self, 'model'):
                self.self_qpos_idx = get_qpos_indices(self.model, controlled_joints)
                self.self_qvel_idx = get_qvel_indices(self.model, controlled_joints)
                self.self_actuators_idx = get_actuator_indices(self.model, controlled_joints)
            else:
                self.self_qpos_idx = get_qpos_indices(self.sim.model, controlled_joints)
                self.self_qvel_idx = get_qvel_indices(self.sim.model, controlled_joints)
                self.self_actuators_idx = get_actuator_indices(self.sim.model, controlled_joints)
        else:
            assert self.model.nv == self.model.nu, "if the number of degrees of freedom is different than the number of actuators you must specify the controlled_joints"
            self.sim_qpos_idx = range(self.sim.model.nq)
            self.sim_qvel_idx = range(self.sim.model.nv)
            self.sim_actuators_idx = range(self.sim.model.nu)
            self.sim_joint_idx = range(self.sim.model.nu)
            
            if hasattr(self, 'model'):
                self.self_qpos_idx = range(self.model.nq)
                self.self_qvel_idx = range(self.model.nv)
                self.self_actuators_idx = range(self.model.nu)
            else:
                self.self_qpos_idx = range(self.sim.model.nq)
                self.self_qvel_idx = range(self.sim.model.nv)
                self.self_actuators_idx = range(self.sim.model.nu)

        # # Get the controlled joints
        # if controlled_joints:
        #     self.controlled_joints = get_qpos_indices(sim.model, controlled_joints)
        # else:
        #     self.controlled_joints = range(sim.model.nq)

        # assert sim.model.nu == len(self.controlled_joints), \
        #     "The number of controlled joints ({}) should match the number of actuators in the model ({})".format(
        #     len(self.controlled_joints), sim.model.nu)

        # Scale the action space to the new scaling.
        self.set_velocity = set_velocity
        self.action_scale = action_scale
        low_pos = sim.model.jnt_range[self.sim_joint_idx, 0] / action_scale
        high_pos = sim.model.jnt_range[self.sim_joint_idx, 1] / action_scale

        low_pos[self.sim.model.jnt_limited[self.sim_joint_idx] == 0] = -np.inf
        high_pos[self.sim.model.jnt_limited[self.sim_joint_idx] == 0] = np.inf

        if self.set_velocity:
            low_vel = np.ones_like(low_pos) / action_scale
            high_vel = np.ones_like(low_pos) / action_scale

            low = np.concatenate([low_pos, low_vel])
            high = np.concatenate([high_pos, high_vel])
        else:
            low = low_pos
            high = high_pos

        low *= action_limit
        high *= action_limit

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

        nu = len(self.self_actuators_idx)
        self.qpos_setpoint = action[0:nu]
        if self.set_velocity:
            self.qvel_setpoint = action[nu:2 * nu]

    def get_torque(self):
        '''
        Computes the torques from the setpoints and the current state.
        '''
        torque = self.kp * (
            self.qpos_setpoint - self.sim.data.qpos[self.sim_qpos_idx]
        ) + self.kd * (
            self.qvel_setpoint - self.sim.data.qvel[self.sim_qvel_idx])

        # Add gravity compensation if necessary
        if self.gravity_comp:
            self.gravity_comp_sim.data.qpos[self.self_qpos_idx] = self.sim.data.qpos[self.sim_qpos_idx].copy()
            self.gravity_comp_sim.data.qvel[self.self_qvel_idx] = np.zeros_like(self.self_qvel_idx)
            self.gravity_comp_sim.data.qacc[self.self_qvel_idx] = np.zeros_like(self.self_qvel_idx)
            mujoco_py.functions.mj_inverse(self.model, self.gravity_comp_sim.data)
            torque += self.gravity_comp_sim.data.qfrc_inverse[self.sim_actuators_idx].copy()

        return torque

class RelativePDController(PDController):
    def set_action(self, action):
        action = action * self.action_scale

        nu = len(self.self_actuators_idx)
        # Set the setpoint difference from the current position.
        self.qpos_setpoint = action[0:nu] + \
            self.sim.data.qpos[self.sim_qpos_idx]
        if self.set_velocity:
            self.qvel_setpoint = action[nu:2 * nu]


register_controller(PDController, 'PDController')
register_controller(RelativePDController, 'RelativePDController')
