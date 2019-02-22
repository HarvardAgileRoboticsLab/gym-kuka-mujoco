import os
import random
import numpy as np

from gym_kuka_mujoco.envs import kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat
from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from gym_kuka_mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices


class PushingEnv(kuka_env.KukaEnv):
    
    def __init__(self,
                 *args,
                 obs_scaling=0.1,
                 use_rel_pos_err=False,
                 pos_reward=True,
                 rot_reward=True,
                 pos_vel_reward=False,
                 rot_vel_reward=False,
                 use_ft_sensor=False,
                 **kwargs):
        
        # Store arguments.
        self.use_ft_sensor = use_ft_sensor
        self.obs_scaling = obs_scaling
        self.use_rel_pos_err = use_rel_pos_err
        self.pos_reward = pos_reward
        self.rot_reward = rot_reward
        self.pos_vel_reward = pos_vel_reward
        self.rot_vel_reward = rot_vel_reward
        
        # Resolve the models path based on the hole_id.
        kwargs['model_path'] = kwargs.get('model_path', 'full_pushing_experiment_no_gravity.xml')
        super(PushingEnv, self).__init__(*args, **kwargs)
        
        # Compute good states using inverse kinematics.
        if self.random_target:
            raise NotImplementedError
        
        self.kuka_pos_idx = get_qpos_indices(self.model, ['kuka_joint_{}'.format(i) for i in range(1,8)])
        self.kuka_vel_idx = get_qvel_indices(self.model, ['kuka_joint_{}'.format(i) for i in range(1,8)])
        self.block_pos_idx = get_qpos_indices(self.model, ['block_position'])
        self.block_vel_idx = get_qvel_indices(self.model, ['block_position'])

        self.init_qpos = np.zeros(self.model.nq)
        self.init_qpos[self.kuka_pos_idx] = np.array([ 0.84985144,  0.97250624,  1.83905997,  1.80017142,  1.01155183, -1.2224522, 2.37542027])
        # self.init_qpos[self.kuka_pos_idx] = np.array([-7.25614932e-06,  5.04007949e-01,  9.31413754e-06, -1.80017133e+00, -6.05474878e-06,  8.37413374e-01,  4.95278012e-06])
        # self.init_qpos[self.kuka_pos_idx] = np.array([0, 0, 0, -2, 0, .9, 0])
        self.init_qpos[self.block_pos_idx] = np.array([.7, 0, 1.2, 1, 0, 0, 0])

        self.block_target_position = np.array([.7, 0, 1.2, 1, 0, 0, 0])

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        reward_info = dict()
        reward = 0.

        if self.pos_reward:
            pos_err = self.data.qpos[self.block_pos_idx][:3] - self.block_target_position[:3]
            reward_info['block_pos_reward'] = np.linalg.norm(pos_err)
            reward += reward_info['block_pos_reward']
        if self.rot_reward:
            rot_err = subQuat(self.data.qpos[self.block_pos_idx][3:], self.block_target_position[3:])
            reward_info['block_rot_reward'] = np.linalg.norm(rot_err)
            reward += reward_info['block_rot_reward']
        if self.pos_vel_reward:
            pos_vel = self.data.qvel[self.block_vel_idx[:3]]
            reward_info['block_pos_vel_reward'] = np.linalg.norm(pos_vel)
            reward += reward_info['block_pos_vel_reward']
        if self.rot_vel_reward:
            rot_vel = self.data.qvel[self.block_vel_idx[3:]]
            reward_info['block_rot_vel_reward'] = np.linalg.norm(rot_vel)
            reward += reward_info['block_rot_vel_reward']

        return reward, reward_info

    def _get_info(self):
        info = dict()
        
        pos_err = self.data.qpos[self.block_pos_idx][:3] - self.block_target_position[:3]
        info['block_pos_dist'] = np.linalg.norm(pos_err)

        rot_err = subQuat(self.data.qpos[self.block_pos_idx][3:], self.block_target_position[3:])
        info['block_rot_dist'] = np.linalg.norm(rot_err)
        
        return info

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''
        if not self.initialized:
            # 14 positions and 13 velocities
            obs = np.zeros(27)
            obs[10] = 1.
        else:
            # Return superclass observation.
            obs = super(PushingEnv, self)._get_state_obs()

        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
        else:
            # Compute F/T sensor data
            ft_obs = self.sim.data.sensordata
            obs = obs / self.obs_scaling

        if self.use_ft_sensor:
            obs = np.concatenate([obs, ft_obs])

        return obs

    def _get_target_obs(self):
        # Compute relative position error
        raise NotImplementedError
        if self.use_rel_pos_err:
            pos, rot = forwardKinSite(self.sim, ['hammer_tip','nail_top'])
            pos_obs = pos[0] - pos[1]
            quat_hammer_tip = mat2Quat(rot[0])
            quat_nail_top = mat2Quat(rot[1])
            rot_obs = subQuat(quat_hammer_tip, quat_nail_top)
            return np.concatenate([pos_obs, rot_obs])
        else:
            return np.array([self.data.qpos[self.nail_idx]])

    def _reset_state(self):
        '''
        Reset the robot state and return the observation.
        '''
        qpos = self.init_qpos.copy()
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)

    def _reset_target(self):
        '''
        Resets the hole position
        '''
        raise NotImplementedError