import os
import numpy as np
import mujoco_py
from gym import spaces
from gym_kuka_mujoco.envs import kuka_env
from gym_kuka_mujoco.utils.quaternion import identity_quat, subQuat, quatIntegrate, mat2Quat
from gym_kuka_mujoco.utils.kinematics import forwardKinSite, forwardKinJacobianSite



class RemoteCenterControlledKukaEnv(kuka_env.KukaEnv):
    '''
    A Kuka environment that uses a low-level inverse dynamics controller
    '''

    def __init__(self,
                 kp=None,
                 control_model_path=None,
                 scale_restart=0.1,
                 **kwargs):

        # Initialize control quantities
        # Note: This must be before the super class constructor because
        #   it calls get_torque() which requires these variables to be set
        self.kp = np.array([1, 1, 1, .1, .1, .1])
        self.kv = 0.

        # randomness in the position on restart
        self.scale_restart = scale_restart

        if 'model_path' not in kwargs:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole.xml')
        super(RemoteCenterControlledKukaEnv, self).__init__(**kwargs)
        
        # Create a model for control
        if control_model_path is None:
            control_model_path = 'full_peg_insertion_experiment_no_collision.xml'
        control_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'assets', control_model_path)
        self.model_for_control = mujoco_py.load_model_from_path(control_model_path)

        self.frame_skip = 50  # Control at 10 Hz

        # Set the action space
        # Note: This must be after the super class constructor is called to
        #   overwrite the original action space.
        # let the cartesian setpoint be maximum 10cm away in any direction
        high_pos = np.inf*np.ones(3)
        low_pos = -high_pos
        pos_scale = 0.1*np.ones(3)

        # let the rotation setpoint be maximum 0.5 rad (~30 deg) away on any axis
        high_rot = np.inf*np.ones(3)
        low_rot = -high_rot
        rot_scale = 0.5*np.ones(3)

        low = np.concatenate((low_pos, low_rot))
        high = np.concatenate((high_pos, high_rot))
        self.action_scaling = np.concatenate((pos_scale, rot_scale))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        self.R = np.eye(low.size)

        self.time_limit = 10

    def _update_action(self, action):
        '''
        Set the setpoints.
        '''
        # Hack to allow different sized action space than the super class
        if len(action) != 6:
            self.pos_set = np.zeros(3)
            self.quat_set = identity_quat.copy()
            return

        action = action * self.action_scaling

        dx = action[0:3].astype(np.float64)
        dr = action[3:6].astype(np.float64)

        # let the controller also set the gains
        # self.kp = action[6:12]
        
        pos, mat = forwardKinSite(self.sim, 'peg_tip')
        quat = mat2Quat(mat)

        self.pos_set = pos + dx
        self.quat_set = quatIntegrate(quat, dr)

        self.sim.data.set_mocap_pos('target_frame', self.pos_set)
        self.sim.data.set_mocap_quat('target_frame', self.quat_set)

    def _get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        pos, mat = forwardKinSite(self.sim, 'peg_tip')
        quat = mat2Quat(mat)
        dx = self.pos_set - pos
        dr = subQuat(self.quat_set, quat)
        dframe = np.concatenate((dx,dr))

        # Compute jacobian of the end effector
        jpos, jrot = forwardKinJacobianSite(self.sim, 'peg_tip')
        J = np.vstack((jpos, jrot))

        external_force = J.T.dot(self.kp*dframe) # virtual force on the end effector
        acc_des = -self.kv*self.sim.data.qvel # virtual damping on the joints

        # Compute torques using inverse dynamics
        self.sim.data.qacc[:] = acc_des
        mujoco_py.functions.mj_inverse(self.model_for_control, self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[:]
        return id_torque + external_force

    def _reset_state(self):
        '''
        Reset the robot state and return the observation.
        '''
        qpos = self.scale_restart*self.np_random.uniform(low=self.model.jnt_range[:,0], high=self.model.jnt_range[:,1], size=self.model.nq)
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)