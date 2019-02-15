import os
import numpy as np
from gym import spaces
import mujoco_py

from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from .base_controller import BaseController
from . import register_controller

class InverseDynamicsController(BaseController):
    '''
    An inverse dynamics controller that used PD gains to compute a desired acceleration.
    '''

    def __init__(self,
                 sim,
                 model_path='full_kuka_no_collision_no_gravity.xml',
                 action_scale=1.,
                 action_limit=1.,
                 kp_id=100.,
                 kd_id='auto',
                 controlled_joints=None):
        super(InverseDynamicsController, self).__init__(sim)
        
        # Create a model for control
        model_path = os.path.join(kuka_asset_dir(), model_path)
        self.model = mujoco_py.load_model_from_path(model_path)

        # Hammer-specific problems
        self.controlled_joints = controlled_joints
        if self.controlled_joints is not None:
            self.controlled_joints = [self.model.joint_name2id(joint) for joint in self.controlled_joints]
 
        # Construct the action space.
        # import pdb
        low = -3*np.ones(self.model.nu)* action_limit
        high = 3*np.ones(self.model.nu)* action_limit
        # if not self.controlled_joints:
        #     low = self.model.jnt_range[:, 0]*action_limit
        #     high = self.model.jnt_range[:, 1]*action_limit
        # else:
        #     low = self.model.jnt_range[self.controlled_joints, 0]*action_limit
        #     high = self.model.jnt_range[self.controlled_joints, 1]*action_limit
        # pdb.set_trace()
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        
        # Controller parameters.
        self.action_scale = action_scale
        self.kp_id = kp_id
        if kd_id == 'auto':
            self.kd_id = 2 * np.sqrt(self.kp_id)
        else:
            self.kd_id = kd_id

        # Initialize setpoint.
        self.qpos_set = np.zeros(self.model.nq)
        self.qvel_set = np.zeros(self.model.nq)


    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        if not self.controlled_joints:
            self.qpos_set = self.action_scale * action[:7]
        else:
            self.qpos_set[self.controlled_joints] = self.action_scale * action[:7]

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        # Compute position and velocity errors.
        qpos_err = self.qpos_set - self.sim.data.qpos
        qvel_err = self.qvel_set - self.sim.data.qvel

        # Compute desired acceleration using inner loop PD law.
        if not self.controlled_joints:
            self.sim.data.qacc[:] = self.kp_id * qpos_err + self.kd_id * qvel_err
        else:
            self.sim.data.qacc[self.controlled_joints] = self.kp_id * qpos_err[self.controlled_joints] + self.kd_id * qvel_err[self.controlled_joints]
        mujoco_py.functions.mj_inverse(self.model, self.sim.data)
        
        if not self.controlled_joints:
            id_torque = self.sim.data.qfrc_inverse[:].copy()
        else:
            id_torque = self.sim.data.qfrc_inverse[self.controlled_joints].copy()

        # Sum the torques.
        return id_torque

class RelativeInverseDynamicsController(InverseDynamicsController):
    def set_action(self, action):
        # Set the setpoint difference from the current position.
        if not self.controlled_joints:
            self.qpos_set = self.sim.data.qpos + action[:7]
        else:
            self.qpos_set[self.controlled_joints] = self.sim.data.qpos[self.controlled_joints] + action[:7]

register_controller(InverseDynamicsController, 'InverseDynamicsController')
register_controller(RelativeInverseDynamicsController, 'RelativeInverseDynamicsController')