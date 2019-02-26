import os

import numpy as np
import gym
import mujoco_py

from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from gym_kuka_mujoco.utils.mujoco_utils import kuka_subtree_mass, get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices 
from .base_controller import BaseController
from . import register_controller

class DirectTorqueController(BaseController):
    '''
    A simple controller that takes raw torque actions.
    '''
    def __init__(self,
                 sim,
                 action_scaling=10.,
                 gravity_comp_model_path=None,
                 controlled_joints=None):
        super(DirectTorqueController, self).__init__(sim)

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

                self.self_qpos_idx = get_qpos_indices(self.model, controlled_joints)
                self.self_qvel_idx = get_qvel_indices(self.model, controlled_joints)
                self.self_actuators_idx = get_actuator_indices(self.model, controlled_joints)
            else:
                assert self.model.nv == self.model.nu, "if the number of degrees of freedom is different than the number of actuators you must specify the controlled_joints"
                self.sim_qpos_idx = range(self.model.nq)
                self.sim_qvel_idx = range(self.model.nv)
                self.sim_actuators_idx = range(self.model.nu)
                self.sim_joint_idx = range(self.model.nu)

                self.self_qpos_idx = range(self.model.nq)
                self.self_qvel_idx = range(self.model.nv)
                self.self_actuators_idx = range(self.model.nu)

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
        torque = self.torque.copy()
        
        # Add gravity compensation if necessary
        if self.gravity_comp:
            self.gravity_comp_sim.data.qpos[:] = self.sim.data.qpos[:].copy()
            self.gravity_comp_sim.data.qvel[:] = np.zeros(self.model.nv)
            self.gravity_comp_sim.data.qacc[:] = np.zeros(self.model.nv)
            mujoco_py.functions.mj_inverse(self.model, self.gravity_comp_sim.data)
            torque += self.gravity_comp_sim.data.qfrc_inverse[self.sim_actuators_idx].copy()

        return torque

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