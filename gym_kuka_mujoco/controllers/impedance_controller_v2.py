import os

import numpy as np
from gym import spaces
import mujoco_py

from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from gym_kuka_mujoco.utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat
from gym_kuka_mujoco.utils.kinematics import forwardKinSite, forwardKinJacobianSite
from .base_controller import BaseController
from . import register_controller
from gym_kuka_mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices 


class ImpedanceControllerV2(BaseController):
    '''
    An inverse dynamics controller that used PD gains to compute a desired acceleration.
    '''

    def __init__(self,
                 sim,
                 pos_scale=1.0,
                 rot_scale=0.3,
                 pos_limit=1.0,
                 rot_limit=1.0,
                 model_path='full_kuka_no_collision_no_gravity.xml',
                 site_name='ee_site',
                 stiffness=None,
                 damping='auto',
                 null_space_damping=1.0,
                 controlled_joints=None,
                 in_ee_frame=False):
        super(ImpedanceControllerV2, self).__init__(sim)

        # Create a model for control
        model_path = os.path.join(kuka_asset_dir(), model_path)
        self.model = mujoco_py.load_model_from_path(model_path)

        self.in_ee_frame = in_ee_frame

        # Construct the action space.
        high_pos = pos_limit*np.ones(3)
        low_pos = -high_pos

        high_rot = rot_limit*np.ones(3)
        low_rot = -high_rot

        high = np.concatenate((high_pos, high_rot))
        low = np.concatenate((low_pos, low_rot))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # Controller parameters.
        self.scale = np.ones(6)
        self.scale[:3] *= pos_scale
        self.scale[3:6] *= rot_scale

        self.site_name = site_name
        self.pos_set = np.zeros(3)
        self.quat_set = identity_quat.copy()

        if stiffness is None:
            self.stiffness = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
        else:
            self.stiffness = np.ones(6)*stiffness

        if damping=='auto':
            self.damping = 2*np.sqrt(self.stiffness)
        else:
            self.damping = np.ones(6)*damping

        self.null_space_damping = null_space_damping

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

    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        action = action * self.scale

        dx = action[0:3].astype(np.float64)
        dr = action[3:6].astype(np.float64)

        pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
        quat = mat2Quat(mat)
        
        if self.in_ee_frame:
            dx = mat.dot(dx)

        self.pos_set = pos + dx
        self.quat_set = quatAdd(quat, dr)

    def get_torque(self):
        '''
        Update the impedance control setpoint and compute the torque.
        '''
        # Compute the pose difference.
        pos, mat = forwardKinSite(self.sim, self.site_name, recompute=False)
        quat = mat2Quat(mat)
        dx = self.pos_set - pos
        dr = subQuat(self.quat_set, quat) # Original
        dframe = np.concatenate((dx,dr))

        # Compute generalized forces from a virtual external force.
        jpos, jrot = forwardKinJacobianSite(self.sim, self.site_name, recompute=False)
        J = np.vstack((jpos[:,self.sim_qvel_idx], jrot[:,self.sim_qvel_idx]))
        cartesian_acc_des = self.stiffness*dframe - self.damping*J.dot(self.sim.data.qvel[self.sim_qvel_idx])
        impedance_acc_des = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-6*np.eye(6), cartesian_acc_des))

        # Add damping in the null space of the the Jacobian
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_vel = projection_matrix.dot(self.sim.data.qvel[self.sim_qvel_idx])
        impedance_acc_des += -self.null_space_damping*null_space_vel # null space damping

        # Cancel other dynamics and add virtual damping using inverse dynamics.
        acc_des = np.zeros(self.sim.model.nv)
        acc_des[self.sim_qvel_idx] = impedance_acc_des
        self.sim.data.qacc[:] = acc_des
        mujoco_py.functions.mj_inverse(self.model, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[self.sim_actuators_idx].copy()
        
        return id_torque

register_controller(ImpedanceControllerV2, "ImpedanceControllerV2")