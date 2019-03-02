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
    default_info = {
        "block_pos_dist": -1.0,
        "block_rot_dist": -1.0
    }
    def __init__(self,
                 *args,
                 obs_scaling=0.1,
                 use_rel_pos_err=False,
                 pos_reward=True,
                 rot_reward=True,
                 pos_vel_reward=False,
                 rot_vel_reward=False,
                 peg_tip_height_reward=True,
                 peg_tip_orientation_reward=True,
                 contact_reward=False,
                 use_ft_sensor=False,
                 random_init_pos_file=None,
                 reward_scale=1.0,
                 **kwargs):
        
        # Store arguments.
        self.use_ft_sensor = use_ft_sensor
        self.obs_scaling = obs_scaling
        self.use_rel_pos_err = use_rel_pos_err
        self.pos_reward = pos_reward
        self.rot_reward = rot_reward
        self.pos_vel_reward = pos_vel_reward
        self.rot_vel_reward = rot_vel_reward
        self.peg_tip_height_reward = peg_tip_height_reward
        self.peg_tip_orientation_reward = peg_tip_orientation_reward
        self.contact_reward = contact_reward
        self.reward_scale = reward_scale

        # Resolve the models path based on the hole_id.
        kwargs['model_path'] = kwargs.get('model_path', 'full_pushing_experiment.xml')
        super(PushingEnv, self).__init__(*args, **kwargs)
        
        # Compute good states using inverse kinematics.
        if self.random_target:
            raise NotImplementedError
        
        self.kuka_pos_idx = get_qpos_indices(self.model, ['kuka_joint_{}'.format(i) for i in range(1,8)])
        self.kuka_vel_idx = get_qvel_indices(self.model, ['kuka_joint_{}'.format(i) for i in range(1,8)])
        self.block_pos_idx = get_qpos_indices(self.model, ['block_position'])
        self.block_vel_idx = get_qvel_indices(self.model, ['block_position'])

        # Set the initial conditions
        if random_init_pos_file is None:
            self.init_qpos_kuka = [
            np.array([-0.39644391,  0.90374878,  0.29834325, -1.32769707,  0.32136548 , 1.0627125, -0.37159982])  # positioned to the -y, +x, -z, and oriented pi/6 rad toward the skinny long block
                # np.array([-0.18183141,  0.82372004,  0.12119514, -1.36660597,  0.13291881,  0.97885664, -0.17394202])  # positioned to the -y, +x, -z, and oriented 0.2 rad toward the skinny long block
                # np.array([-3.09971876e-02,  8.27156181e-01, -3.99097887e-04, -1.33895589e+00, 3.54769752e-04,  9.75480647e-01, -3.14663096e-02]) # positioned to the -y, +x, -z of the skinny long block
                # np.array([-3.08291276e-02,  8.00410366e-01, -6.76915982e-04, -1.33038224e+00, 5.73359368e-04, 1.01080023e+00, -3.16050986e-02]), # positioned to the -y and +x of the skinny long block
                # np.array([ 2.42308236, -1.01571971, 1.13996587, 1.56773622, 2.17062176, 1.20969055, 0.61193454]), # positioned to the -y and +x of the skinny block
                # np.array([-0.07025654, 0.6290658, -0.00323965, -1.64794655, 0.0025054, 0.86458435, -0.07450195]), # positioned to the -y and +x of the block
                # np.array([ 0.74946844, 0.98614739, 1.88508577, 1.80629075, 1.02973813, -1.18159247, 2.28928049]), # positioned to the -y of the block
                # np.array([ 0.84985144, 0.97250624, 1.83905997, 1.80017142, 1.01155183, -1.2224522, 2.37542027]), # positioned above the block (bent elbow)
                # np.array([-7.25614932e-06,  5.04007949e-01,  9.31413754e-06, -1.80017133e+00, -6.05474878e-06,  8.37413374e-01,  4.95278012e-06]), # positioned above the block (straight elbow)
            ]
        else:
            random_init_pos_file = os.path.join(kuka_asset_dir(), random_init_pos_file)
            self.init_qpos_kuka = np.load(random_init_pos_file)

        self.init_qpos_block = np.array([.7, 0, 1.2, 1, 0, 0, 0])
        self.table_height = self.init_qpos_block[2] - 0.02
        # import pdb; pdb.set_trace()

        # self.block_target_position = np.array([.7, .1, 1.2, 0.70710678118, 0, 0, 0.70710678118])
        # self.block_target_position = np.array([.7, .1, 1.2, 0., 0., 0., 1.])
        self.block_target_position = np.array([0.7, 0.1, 1.2, 0.92387953251, 0, 0, 0.38268343236])
        self.tip_geom_id = self.model.geom_name2id('peg_tip_geom')
        self.block_geom_id = self.model.geom_name2id('block_geom')

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        reward_info = dict()
        reward = 0.

        if self.pos_reward:
            pos_err = self.data.qpos[self.block_pos_idx][:3] - self.block_target_position[:3]
            reward_info['block_pos_reward'] = -np.linalg.norm(pos_err)*10
            reward += reward_info['block_pos_reward']
        if self.rot_reward:
            rot_err = subQuat(self.data.qpos[self.block_pos_idx][3:], self.block_target_position[3:])
            reward_info['block_rot_reward'] = -np.linalg.norm(rot_err)
            reward += reward_info['block_rot_reward']
        if self.pos_vel_reward:
            raise NotImplementedError
            # TODO: this should give positive reward when moving towards the goal and negative reward when moving away from the goal
            pos_vel = self.data.qvel[self.block_vel_idx[:3]]
            reward_info['block_pos_vel_reward'] = -np.linalg.norm(pos_vel)
            reward += reward_info['block_pos_vel_reward']
        if self.rot_vel_reward:
            raise NotImplementedError
            # TODO: this should give positive reward when moving towards the goal and negative reward when moving away from the goal
            rot_vel = self.data.qvel[self.block_vel_idx[3:]]
            reward_info['block_rot_vel_reward'] = -np.linalg.norm(rot_vel)
            reward += reward_info['block_rot_vel_reward']
        if self.peg_tip_height_reward or self.peg_tip_orientation_reward:
            tip_pos, tip_rot = forwardKinSite(self.sim, ['peg_tip'])
            tip_pos = tip_pos[0]
            tip_rot = tip_rot[0]
        if self.peg_tip_height_reward:
            tip_height_err = tip_pos[2] - self.table_height
            reward_info['peg_tip_height_reward'] = -tip_height_err**2
            reward += reward_info['peg_tip_height_reward']
        if self.peg_tip_orientation_reward:
            tip_quat = mat2Quat(tip_rot)
            tip_rot_err = subQuat(tip_quat, np.array([0., 1., 0., 0.]))
            reward_info['peg_tip_orientation_reward'] = -(np.linalg.norm(tip_rot_err)/10)**2
            reward += reward_info['peg_tip_orientation_reward']
        if self.contact_reward:
            contacts = self.sim.data.contact[:self.sim.data.ncon]
            contact = 0.
            for c in contacts:
                geoms = c.geom1, c.geom2
                if (self.tip_geom_id in geoms) and (self.block_geom_id in geoms):
                    contact = 1.
                    break
            reward_info['contact_reward'] = contact*.1
            reward += reward_info['contact_reward']
        
        return reward*self.reward_scale, reward_info

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

        if not self.initialized:
            ee_lin_vel_obs = np.zeros(3)
            ee_rot_vel_obs = np.zeros(3)
        else:
            peg_tip_id = self.model.site_name2id('peg_tip')
            jacp, jacr = forwardKinJacobianSite(self.sim, peg_tip_id, recompute=False)
            ee_lin_vel_obs = jacp.dot(self.sim.data.qvel)
            ee_rot_vel_obs = jacr.dot(self.sim.data.qvel)
        
        obs = np.concatenate([ee_lin_vel_obs, ee_rot_vel_obs])

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
        # Randomly choose the kuka initial position
        qpos = np.zeros(self.model.nq)
        qpos[self.kuka_pos_idx] = self.init_qpos_kuka[self.np_random.choice(len(self.init_qpos_kuka))].copy()
        # qpos[self.kuka_pos_idx] = self.init_qpos_kuka
        qpos[self.block_pos_idx] = self.init_qpos_block.copy()
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)

    def _reset_target(self):
        '''
        Resets the hole position
        '''
        raise NotImplementedError