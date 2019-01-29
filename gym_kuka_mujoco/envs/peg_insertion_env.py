import random
import numpy as np

from gym_kuka_mujoco.envs import id_controlled_kuka_env
from gym_kuka_mujoco.envs import remote_center_controlled_kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat

class PegInsertionEnv(id_controlled_kuka_env.DiffIdControlledKukaEnv):
    sample_good_states = True
    use_ft_sensor = False
    use_rel_pos_err = False
    # Cost parameters
    regularize_pose = True
    quadratic_cost = True
    linear_cost = False
    logarithmic_cost = False
    sparse_cost = False

    info_keywords = ('tip_distance',)
    
    def __init__(self, *args, hole_id=99, **kwargs):
        if hole_id >= 0:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_moving_hole_id={:03d}.xml'.format(hole_id))
        else:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole.xml')       
        kwargs['control_model_path'] = kwargs.get('control_model_path', 'full_peg_insertion_experiment_no_collision.xml')
        self.fine_scaling = .1
        super(PegInsertionEnv, self).__init__(*args, **kwargs)

        self.time_limit = 2
        self.Q_pos = np.diag([100,100,100])
        self.Q_rot = np.diag([1,1,1])
        if self.regularize_pose:
            self.Q_pose_reg = np.eye(7)
        # self.Q_vel = np.diag([1,1,.1])
        # self.eps = 1e-2



        if self.sample_good_states:
            self.good_states = hole_insertion_samples(self.sim, range=[0.,0.06])

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        # compute position and rotation error
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        pos_err = pos[0] - pos[1]
        dist = np.sqrt(pos_err.dot(pos_err))
        peg_quat = mat2Quat(rot[0])
        hole_quat = mat2Quat(rot[1])
        rot_err = subQuat(peg_quat, hole_quat)

        pose_err = self.sim.data.qpos - self.good_states[0]
        
        peg_tip_id = self.model.site_name2id('peg_tip')
        jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id)
        peg_tip_vel = jacp.dot(self.data.qvel[:])
        # print("reward_dist: {}".format(dist))
        
        # quadratic cost on the error and action
        # rotate the cost terms to align with the hole
        Q_pos = rotate_cost_by_matrix(self.Q_pos,rot[1].T)
        # Q_vel = rotate_cost_by_matrix(self.Q_vel,rot[1].T)
        Q_rot = self.Q_rot

        reward_info = dict()
        reward = 0.

        # reward_info['quaternion_reward'] = -rot_err.dot(Q_rot).dot(rot_err)
        
        if self.quadratic_cost:
            reward_info['quadratic_position_reward'] = -pos_err.dot(Q_pos).dot(pos_err)
            reward += reward_info['quadratic_position_reward']

        if self.linear_cost:
            reward_info['linear_position_reward'] = -np.sqrt(pos_err.dot(Q_pos).dot(pos_err))
            reward += reward_info['linear_position_reward']

        if self.logarithmic_cost:
            rew_scale = 2
            eps = 10.0**(-rew_scale)
            zero_crossing = 0.05
            reward_info['logarithmic_position_reward'] = -np.log10(np.sqrt(pos_err.dot(Q_pos).dot(pos_err))/zero_crossing*(1-eps) + eps)
            reward += reward_info['logarithmic_position_reward']

        if self.sparse_cost:
            reward_info['sparse_position_reward'] = 10.0 if np.sqrt(pos_err.dot(pos_err)) < 1e-2 else 0
            reward += reward_info['sparse_position_reward']

        if self.regularize_pose:
            reward_info['pose_regularizer_reward'] = -pose_err.dot(self.Q_pose_reg).dot(pose_err)
            reward += reward_info['pose_regularizer_reward']
        
        # reward_info['velocity_reward'] = -np.sqrt(peg_tip_vel.dot(Q_vel).dot(peg_tip_vel)) 
        # reward += reward_info['velocity_reward']

        return reward, reward_info

    def _get_info(self):
        info = dict()
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        pos_err = pos[0] - pos[1]
        dist = np.sqrt(pos_err.dot(pos_err))
        info['tip_distance'] = dist
        return info

    def _update_action(self, action):
        action = action * self.fine_scaling
        super(PegInsertionEnv, self)._update_action(action)

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''

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

            obs[:7] = obs[:7]
            obs = obs / self.fine_scaling

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
        if self.sample_good_states and self.np_random.uniform() < 0*0.5:
            qpos = self.np_random.choice(self.good_states)
        else:
            qpos = self.good_states[-1] + self.np_random.uniform(-.01,.01,7)
        
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)


class QuadraticCostPegInsertionEnv(PegInsertionEnv):
    # Cost parameters
    regularize_pose = False
    quadratic_cost = True
    linear_cost = False
    logarithmic_cost = False
    sparse_cost = False

    def __init__(self, *args, hole_id=-1, **kwargs):
        super(QuadraticCostPegInsertionEnv, self).__init__(*args, hole_id=hole_id, **kwargs)

class LinearCostPegInsertionEnv(PegInsertionEnv):
    # Cost parameters
    regularize_pose = False
    quadratic_cost = False
    linear_cost = True
    logarithmic_cost = False
    sparse_cost = False

    def __init__(self, *args, hole_id=-1, **kwargs):
        super(LinearCostPegInsertionEnv, self).__init__(*args, hole_id=hole_id, **kwargs)

class QuadraticLogarithmicCostPegInsertionEnv(PegInsertionEnv):
    # Cost parameters
    regularize_pose = False
    quadratic_cost = True
    linear_cost = False
    logarithmic_cost = True
    sparse_cost = False

    def __init__(self, *args, hole_id=-1, **kwargs):
        super(QuadraticLogarithmicCostPegInsertionEnv, self).__init__(*args, hole_id=hole_id, **kwargs)

class QuadraticSparseCostPegInsertionEnv(PegInsertionEnv):
    # Cost parameters
    regularize_pose = False
    quadratic_cost = True
    linear_cost = False
    logarithmic_cost = False
    sparse_cost = True

    def __init__(self, *args, hole_id=-1, **kwargs):
        super(QuadraticSparseCostPegInsertionEnv, self).__init__(*args, hole_id=hole_id, **kwargs)

class QuadraticRegularizedCostPegInsertionEnv(PegInsertionEnv):
    # Cost parameters
    regularize_pose = True
    quadratic_cost = True
    linear_cost = False
    logarithmic_cost = False
    sparse_cost = False

    def __init__(self, *args, hole_id=-1, **kwargs):
        super(QuadraticRegularizedCostPegInsertionEnv, self).__init__(*args, hole_id=hole_id, **kwargs)















































class RemoteCenterPegInsertionEnv(remote_center_controlled_kuka_env.RemoteCenterControlledKukaEnv):
    setpoint_diff = True
    sample_good_states = True
    use_ft_sensor = True
    # hole_size = "tight"
    
    def __init__(self, *args, hole_id=None, **kwargs):
        print("RemoteCenterPegInsertionEnv __init__()")
        if hole_id is None:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment.xml')
        elif hole_id >= 0:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_moving_hole_id={:03d}.xml'.format(hole_id))
        else:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole.xml')       
        kwargs['control_model_path'] = kwargs.get('control_model_path', 'full_peg_insertion_experiment_no_collision.xml')
        super(RemoteCenterPegInsertionEnv, self).__init__(*args, **kwargs)

        self.time_limit = 3
        self.Q_pos = np.diag([100,100,10])
        self.Q_rot = np.diag([1,1,1])
        self.Q_vel = np.diag([1,1,.1])
        self.eps = 1e-2


        if self.sample_good_states:
            self.good_states = hole_insertion_samples(self.sim, range=[0.,0.06])

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        # compute position and rotation error
        pos, rot, pos_err, rot_err = self.get_kinematic_data()
        dist = np.sqrt(pos_err.dot(pos_err))

        peg_tip_id = self.model.site_name2id('peg_tip')
        jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id)
        peg_tip_vel = jacp.dot(self.data.qvel[:])
        # print("reward_dist: {}".format(dist))
        reward_info = dict()

        # quadratic cost on the error and action
        # rotate the cost terms to align with the hole
        Q_pos = rotate_cost_by_matrix(self.Q_pos,rot[1].T)
        Q_vel = rotate_cost_by_matrix(self.Q_vel,rot[1].T)
        Q_rot = self.Q_rot

        reward_info['position_reward'] = -np.sqrt(pos_err.dot(Q_pos).dot(pos_err))
        # reward_info['quaternion_reward'] = -np.sqrt(rot_err.dot(Q_rot).dot(rot_err))
        # reward_info['velocity_reward'] = -np.sqrt(peg_tip_vel.dot(Q_vel).dot(peg_tip_vel)) 
        reward = reward_info['position_reward']
        # reward += reward_info['velocity_reward']
        #reward -= action.dot(self.R).dot(action)
        # reward += 10.0 if dist < self.eps else 0.0
        # reward -= 100.0 if dist > .2 else 0.0

        return reward, reward_info

    def get_kinematic_data(self):
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        pos_err = pos[0] - pos[1]
        dist = np.sqrt(pos_err.dot(pos_err))
        peg_quat = mat2Quat(rot[0])
        hole_quat = mat2Quat(rot[1])
        rot_err = subQuat(peg_quat, hole_quat)

        return pos, rot, pos_err, rot_err

    def _get_done(self):
        # Don't terminate early
        return False

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''

        # Return superclass observation.
        obs = super(RemoteCenterPegInsertionEnv, self)._get_state_obs()

        pos, quat, pos_err, quat_err = self.get_kinematic_data()
        obs = np.concatenate([obs, pos_err])
        return obs
        # obs = np.concatenate([obs, quat_err])

        if not self.use_ft_sensor:
            return obs    

        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
        else:
            ft_obs = self.sim.data.sensordata

        obs = np.concatenate([obs, ft_obs])
        return obs

    def _reset_state(self):
        '''
        Reset the robot state and return the observation.
        '''
        qpos = self.good_states[-1] + self.np_random.uniform(-.1,.1,7)
        
        # qpos = np.zeros(7)
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)
