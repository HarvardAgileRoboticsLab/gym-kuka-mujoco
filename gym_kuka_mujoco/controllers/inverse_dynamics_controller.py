import os
import numpy as np
from gym import spaces
import mujoco_py

from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from .base_controller import BaseController
from . import register_controller
from gym_kuka_mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices 

class InverseDynamicsController(BaseController):
    '''
    An inverse dynamics controller that used PD gains to compute a desired acceleration.
    '''

    def __init__(self,
                 sim,
                 model_path='full_kuka_no_collision_no_gravity.xml',
                 action_scale=1.0,
                 action_limit=1.0,
                 kp_id=1.0,
                 kd_id='auto',
                 controlled_joints=None,
                 set_velocity=False,
                 keep_finite=False):
        super(InverseDynamicsController, self).__init__(sim)
        
        # Create a model for control
        model_path = os.path.join(kuka_asset_dir(), model_path)
        self.model = mujoco_py.load_model_from_path(model_path)
        assert self.model.nq == sim.model.nq, "the number of states in the controlled model and the simulated model must be the same"

        self.set_velocity = set_velocity

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
 
        low = self.sim.model.jnt_range[self.sim_joint_idx, 0]
        high = self.sim.model.jnt_range[self.sim_joint_idx, 1]

        low[self.sim.model.jnt_limited[self.sim_joint_idx] == 0] = -np.inf
        high[self.sim.model.jnt_limited[self.sim_joint_idx] == 0] = np.inf
        
        if keep_finite:
            # Don't allow infinite bounds (necessary for SAC)
            low[not np.isfinite(low)] = -3.
            high[not np.isfinite(high)] = 3.

        low = low*action_limit
        high = high*action_limit

        self.action_space = spaces.Box(low, high, dtype=np.float32)
        
        # Controller parameters.
        self.action_scale = action_scale
        self.kp_id = kp_id
        if kd_id == 'auto':
            self.kd_id = 2 * np.sqrt(self.kp_id)
        else:
            self.kd_id = kd_id

        # Initialize setpoint.
        self.sim_qpos_set = sim.data.qpos[self.sim_qpos_idx].copy()
        self.sim_qvel_set = np.zeros(len(self.sim_qvel_idx))


    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        nq = len(self.sim_qpos_idx)
        nv = len(self.sim_qvel_idx)

        self.sim_qpos_set = self.action_scale * action[nq]
        if self.set_velocity:
            self.sim_qvel_set = self.action_scale * action[nq:nv]

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        # Compute position and velocity errors.
        qpos_err = self.sim_qpos_set - self.sim.data.qpos[self.sim_qpos_idx]
        qvel_err = self.sim_qvel_set - self.sim.data.qvel[self.sim_qvel_idx]

        # Compute desired acceleration using inner loop PD law.
        qacc_des = np.zeros(self.sim.model.nv)
        qacc_des[self.sim_qvel_idx] = self.kp_id * qpos_err + self.kd_id * qvel_err
        
        # Compute the inverse dynamics.
        self.sim.data.qacc[:] = qacc_des.copy()
        mujoco_py.functions.mj_inverse(self.model, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[self.sim_actuators_idx].copy()

        # Sum the torques.
        return id_torque

class RelativeInverseDynamicsController(InverseDynamicsController):
    def set_action(self, action):
        nq = len(self.sim_qpos_idx)
        nv = len(self.sim_qvel_idx)

        # Set the setpoint difference from the current position.
        self.sim_qpos_set = self.sim.data.qpos[self.sim_qpos_idx] + self.action_scale * action[:nq]
        if self.set_velocity:
            self.sim_qvel_set = self.action_scale * action[nq:nv]

register_controller(InverseDynamicsController, 'InverseDynamicsController')
register_controller(RelativeInverseDynamicsController, 'RelativeInverseDynamicsController')