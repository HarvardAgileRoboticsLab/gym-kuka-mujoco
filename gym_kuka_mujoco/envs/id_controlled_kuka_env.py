import os
import numpy as np
import mujoco_py
from gym import spaces
from gym_kuka_mujoco.envs import kuka_env


class IdControlledKukaEnv(kuka_env.KukaEnv):
    '''
    A Kuka environment that uses a low-level inverse dynamics controller
    '''
    setpoint_diff = False

    def __init__(self,
                 kp_id=None,
                 kd_id=None,
                 kp_pd=None,
                 kd_pd=None,
                 **kwargs):

        # Initialize control quantities
        # Note: This must be before the super class constructor because
        #   it calls get_torque() which requires these variables to be set
        self.qpos_set = np.zeros(7)
        self.qvel_set = np.zeros(7)
        self.kp_id = np.zeros(7)
        self.kd_id = np.zeros(7)
        self.kp_pd = np.zeros(7)
        self.kd_pd = np.zeros(7)

        super(IdControlledKukaEnv, self).__init__(**kwargs)
        
        # Create a model for control
        model_filename = 'full_kuka_no_collision.xml'
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'assets', model_filename)
        self.model_for_control = mujoco_py.load_model_from_path(model_path)


        self.frame_skip = 50  # Control at 10 Hz

        # Set the controller gains
        # Note: This must be after the super class constructor is called
        #   because we need access to the model.
        self.kp_id = kp_id if kp_id is not None else 100
        self.kd_id = kd_id if kd_id is not None else 2 * np.sqrt(self.kp_id)
        self.kp_pd = kp_pd if kp_pd is not None else \
                    0*1e-2*self.subtree_mass()
        self.kd_pd = kd_pd if kd_pd is not None else \
                    0*np.minimum(self.kp_pd*self.model.opt.timestep,
                        2*np.sqrt(self.subtree_mass()*self.kp_pd))

        # Set the action and observation spaces
        # Note: This must be after the super class constructor is called to
        #   overwrite the original action space.
        low_pos = self.model.jnt_range[:, 0]
        high_pos = self.model.jnt_range[:, 1]

        low_vel = -3 * np.ones(self.model.nv)
        high_vel = 3 * np.ones(self.model.nv)

        low = .1 * np.concatenate((low_pos, low_vel))
        high = .1 * np.concatenate((high_pos, high_vel))
        self.action_space = spaces.Box(high, low, dtype=np.float32)
        self.observation_space = spaces.Box(high, low, dtype=np.float32)

        # Overwrite the action cost.
        self.state_des = np.zeros(14)
        self.Q = 1e-2 * np.eye(14)
        self.R = 1e-6 * np.eye(14)
        self.eps = 1e-1

    def subtree_mass(self):
        '''
        Compute the subtree mass of the Kuka Arm using the actual link names.
        '''
        body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
        body_ids = [self.model.body_name2id(n) for n in body_names]
        return self.model.body_subtreemass[body_ids]

    def update_action(self, action):
        '''
        Set the setpoints.
        '''
        # Hack to allow different sized action space than the super class
        if len(action) != self.model.nq + self.model.nv:
            self.qpos_set = np.zeros(self.model.nq)
            self.qvel_set = np.zeros(self.model.nv)
            return

        # Check if we are setting the action space or small differences
        # from the action space.
        if self.setpoint_diff:
            # Scale to encourage only small differences from the current
            # setpoint
            self.qpos_set = self.sim.data.qpos + 1e-2*action[:7]
            self.qvel_set = self.sim.data.qvel + 1e-2*action[7:14]
            self.qvel_set = np.zeros(7)
        else:
            # Set the PD setpoint directly.
            self.qpos_set = action[:7]
            self.qvel_set = action[7:14]

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        # Compute position and velocity errors
        qpos_err = self.qpos_set - self.sim.data.qpos
        qvel_err = self.qvel_set - self.sim.data.qvel

        # Compute desired acceleration using inner loop PD law
        self.sim.data.qacc[:] = self.kp_id * qpos_err + self.kd_id * qvel_err
        mujoco_py.functions.mj_inverse(self.model_for_control, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[:] / 300

        # Compute torque from outer loop PD law
        pd_torque = self.kp_pd * qpos_err / 300 + self.kd_pd * qvel_err / 300

        # Sum the torques
        return id_torque + pd_torque


class DiffIdControlledKukaEnv(IdControlledKukaEnv):
    setpoint_diff = True
