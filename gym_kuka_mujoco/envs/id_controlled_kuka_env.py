import os
import numpy as np
import mujoco_py
from gym import spaces
from gym_kuka_mujoco.envs import kuka_env


class IdControlledKukaEnv(kuka_env.KukaEnv):
    '''
    A Kuka environment that uses a low-level inverse dynamics controller.
    '''
    def __init__(self,
                 kp_id=None,
                 kd_id=None,
                 control_model_path=None,
                 **kwargs):

        # Initialize control quantities
        # Note: This must be before the super class constructor because
        #   it calls get_torque() which requires these variables to be set
        self.qpos_set = np.zeros(7)
        self.qvel_set = np.zeros(7)

        super(IdControlledKukaEnv, self).__init__(**kwargs)
        
        # Create a model for control
        if control_model_path is None:
            control_model_path = 'full_kuka_no_collision_no_gravity.xml'
        control_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'assets', control_model_path)
        self.model_for_control = mujoco_py.load_model_from_path(control_model_path)

        self.frame_skip = 50  # Control at 10 Hz

        # Set the controller gains
        # Note: This must be after the super class constructor is called
        #   because we need access to the model.
        self.kp_id = kp_id if kp_id is not None else 100
        self.kd_id = kd_id if kd_id is not None else 2 * np.sqrt(self.kp_id)
        
        # Set the action space
        # Note: This must be after the super class constructor is called to
        #   overwrite the original action space.
        low = self.model.jnt_range[:, 0]
        high = self.model.jnt_range[:, 1]
        
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        # Overwrite the action cost.
        self.state_des = np.zeros(14)

    def _update_action(self, action):
        '''
        Set the setpoint.
        '''
        self.qpos_set = action[:7]

    def _get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        # Compute position and velocity errors
        qpos_err = self.qpos_set - self.sim.data.qpos
        qvel_err = self.qvel_set - self.sim.data.qvel

        # Compute desired acceleration using inner loop PD law
        self.sim.data.qacc[:] = self.kp_id * qpos_err + self.kd_id * qvel_err
        mujoco_py.functions.mj_inverse(self.model_for_control, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[:]

        # Sum the torques
        return id_torque

class DiffIdControlledKukaEnv(IdControlledKukaEnv):
    def _update_action(self, action):
        # Set the setpoint difference from the current position.
        self.qpos_set = self.sim.data.qpos + action[:7]