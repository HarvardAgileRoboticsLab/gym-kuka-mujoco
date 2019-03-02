import os
import random
import numpy as np

from gym_kuka_mujoco.envs import kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat
from gym_kuka_mujoco.envs.assets import kuka_asset_dir

class HammerEnv(kuka_env.KukaEnv):
    
    def __init__(self,
                 *args,
                 obs_scaling=0.1,
                 use_ft_sensor=False,
                 use_rel_pos_err=False,
                 pos_reward=True,
                 vel_reward=False,
                 sac_reward_scale=1.0,
                 **kwargs):
        # Store arguments.
        self.obs_scaling = obs_scaling
        self.use_ft_sensor = use_ft_sensor
        self.use_rel_pos_err = use_rel_pos_err
        self.sac_reward_scale = sac_reward_scale
        
        # Resolve the models path based on the hole_id.
        kwargs['model_path'] = kwargs.get('model_path', 'full_hammer_experiment_no_gravity.xml')
        super(HammerEnv, self).__init__(*args, **kwargs)
        
        # Compute good states using inverse kinematics.
        if self.random_target:
            raise NotImplementedError
        
        self.kuka_idx = [self.model.joint_name2id('kuka_joint_{}'.format(i)) for i in range(1,8)]
        self.nail_idx = self.model.joint_name2id('nail_position')
        self.init_qpos = np.zeros(8)
        self.init_qpos[self.kuka_idx] = np.array([0, -.5, 0, -2, 0, 0, 0])
        self.pos_reward = pos_reward
        self.vel_reward = vel_reward

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        reward_info = dict()
        reward = 0.

        if self.pos_reward:
            reward_info['nail_pos_reward'] = -self.data.qpos[self.nail_idx]
            reward += reward_info['nail_pos_reward']
        if self.vel_reward:
            reward_info['nail_vel_reward'] = -self.data.qvel[self.nail_idx]
            reward += reward_info['nail_vel_reward']

        return reward*self.sac_reward_scale, reward_info # *100 for SAC

    def _get_info(self):
        info = dict()
        info['nail_depth'] = -self.data.qpos[self.nail_idx]
        return info

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''
        if not self.initialized:
            obs = np.zeros(16)
        else:
            # Return superclass observation.
            obs = super(HammerEnv, self)._get_state_obs()

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
        qvel = np.zeros(8)
        self.set_state(qpos, qvel)

    def _reset_target(self):
        '''
        Resets the hole position
        '''
        raise NotImplementedError