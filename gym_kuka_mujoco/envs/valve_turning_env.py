import os
import random
import numpy as np

from gym_kuka_mujoco.envs import kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat
from gym_kuka_mujoco.envs.assets import kuka_asset_dir

class PegInsertionEnv(kuka_env.KukaEnv):
    
    def __init__(self,
                 *args,
                 gravity=True,
                 obs_scaling=0.1,
                 use_ft_sensor=False,
                 use_rel_pos_err=False,
                 quadratic_cost=True,
                 linear_cost=False,
                 **kwargs):
        
        # Store arguments.
        self.obs_scaling = obs_scaling
        self.use_ft_sensor = use_ft_sensor
        self.use_rel_pos_err = use_rel_pos_err
        self.quadratic_cost = quadratic_cost
        self.linear_cost = linear_cost
        
        # Resolve the models path based on the hole_id.
        gravity_string = '' if gravity else '_no_gravity'
        kwargs['model_path'] = kwargs.get('model_path', 'full_valve_turning_experiment{}.xml'.format(gravity_string))
        super(PegInsertionEnv, self).__init__(*args, **kwargs)
        

        # Compute good states using inverse kinematics.
        if self.random_target:
            raise NotImplementedError
            self._reset_target()

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        raise NotImplementedError

        if self.quadratic_cost:
            reward_info['quadratic_position_reward'] = -pos_err.dot(Q_pos).dot(pos_err)
            reward += reward_info['quadratic_position_reward']

        if self.linear_cost:
            reward_info['linear_position_reward'] = -np.sqrt(pos_err.dot(Q_pos).dot(pos_err))
            reward += reward_info['linear_position_reward']

        return reward, reward_info

    def _get_info(self):
        raise NotImplementedError
        info = dict()
        return info

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''

        raise NotImplementedError

        # Return superclass observation.
        obs = super(PegInsertionEnv, self)._get_state_obs()


        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
            pos_err = np.zeros(3)
        else:
            # Compute F/T sensor data
            ft_obs = self.sim.data.sensordata
            
            # Compute relative position error
            pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'])
            pos_err = pos[0] - pos[1]

            obs = obs / self.obs_scaling

        if self.use_ft_sensor:
            obs = np.concatenate([obs, ft_obs])

        if self.use_rel_pos_err:
            obs = np.concatenate([obs, pos_err])

        return obs

    def _get_target_obs(self):
        raise NotImplementedError

    def _reset_state(self):
        '''
        Reset the robot state and return the observation.
        '''

        raise NotImplementedError

        qpos = self.initial_pose + self.np_random.uniform(-.01,.01,7)
        
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)

    def _reset_target(self):
        '''
        Resets the hole position
        '''
        raise NotImplementedError
        hole_data = self.np_random.choice(self.reachable_holes)
        self.good_states = hole_data['good_poses']
        self.sim.data.set_mocap_pos('hole', hole_data['hole_pos'])
        self.sim.data.set_mocap_quat('hole', hole_data['hole_quat'])
