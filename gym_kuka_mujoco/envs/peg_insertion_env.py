import os
import random
import numpy as np

from gym_kuka_mujoco.envs import kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat, quatAdd
from gym_kuka_mujoco.envs.assets import kuka_asset_dir

class PegInsertionEnv(kuka_env.KukaEnv):
    
    def __init__(self,
                 *args,
                 hole_id=99,
                 gravity=True,
                 obs_scaling=0.1,
                 sample_good_states=False,
                 use_ft_sensor=False,
                 use_rel_pos_err=False,
                 quadratic_cost=True,
                 quadratic_rot_cost=True,
                 regularize_pose=False,
                 linear_cost=False,
                 logarithmic_cost=False,
                 sparse_cost=False,
                 observe_joints=True,
                 in_peg_frame=False,
                 random_hole_file='random_reachable_holes_small_randomness.npy',
                 init_randomness=0.01,
                 **kwargs):
        
        # Store arguments.
        self.obs_scaling = obs_scaling
        self.sample_good_states = sample_good_states
        self.use_ft_sensor = use_ft_sensor
        self.use_rel_pos_err = use_rel_pos_err
        self.regularize_pose = regularize_pose
        self.quadratic_cost = quadratic_cost
        self.linear_cost = linear_cost
        self.logarithmic_cost = logarithmic_cost
        self.sparse_cost = sparse_cost
        self.quadratic_rot_cost = quadratic_rot_cost
        self.observe_joints = observe_joints
        self.in_peg_frame = in_peg_frame
        self.init_randomness = init_randomness
        
        # Resolve the models path based on the hole_id.
        gravity_string = '' if gravity else '_no_gravity'
        if hole_id >= 0:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment{}_moving_hole_id={:03d}.xml'.format(gravity_string, hole_id))
        else:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole{}.xml').format(gravity_string)       
        super(PegInsertionEnv, self).__init__(*args, **kwargs)
        

        self.Q_pos = np.diag([100,100,100])
        self.Q_rot = np.diag([30,30,30])
        if self.regularize_pose:
            self.Q_pose_reg = np.eye(7)

        # Compute good states using inverse kinematics.
        if self.random_target:
            self.reachable_holes = np.load(os.path.join(kuka_asset_dir(), random_hole_file))
            self._reset_target()
        else:
            self.good_states = hole_insertion_samples(self.sim, range=[0.,0.06])

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        # compute position and rotation error
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'], recompute=False)
        pos_err = pos[0] - pos[1]
        dist = np.sqrt(pos_err.dot(pos_err))
        peg_quat = mat2Quat(rot[0])
        hole_quat = mat2Quat(rot[1])
        rot_err = subQuat(peg_quat, hole_quat)

        pose_err = self.sim.data.qpos - self.good_states[0]
        
        peg_tip_id = self.model.site_name2id('peg_tip')
        jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id, recompute=False)
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
        if self.quadratic_rot_cost:
            reward_info['quadratic_orientation_reward'] = -rot_err.dot(Q_rot).dot(rot_err)
            reward += reward_info['quadratic_orientation_reward']

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
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'], recompute=False)
        pos_err = pos[0] - pos[1]
        dist = np.sqrt(pos_err.dot(pos_err))
        info['tip_distance'] = dist
        info['success'] = float(dist < 1e-2)
        info['qpos'] = self.data.qpos.copy()
        info['qvel'] = self.data.qvel.copy()
        info['action'] = self.last_action
        return info

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''

        # Return superclass observation.
        if self.observe_joints:
            obs = super(PegInsertionEnv, self)._get_state_obs()
        else:
            obs = np.zeros(0)

        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
            pos_err = np.zeros(3)
        else:
            # Compute F/T sensor data
            ft_obs = self.sim.data.sensordata
            

            obs = obs / self.obs_scaling

        if self.use_ft_sensor:
            obs = np.concatenate([obs, ft_obs])

        return obs

# def test_quatInegrate():
#     q2 = random_quat()
#     v = subQuat(q2, identity_quat)
#     q2_ = quatIntegrate(identity_quat, v)
#     assert np.allclose(q2, q2_), 'quatIntegrate test failed'

    def _get_target_obs(self):
        # Compute relative position error
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base','hole_top'])
        peg_tip_id = self.model.site_name2id('peg_tip')
        jacp, jacr = forwardKinJacobianSite(self.sim, peg_tip_id, recompute=False)
        
        if self.use_rel_pos_err:
            pos_obs = pos[1] - pos[0]
            quat_peg_tip = mat2Quat(rot[0])
            quat_hole_base = mat2Quat(rot[1])
            rot_obs = subQuat(quat_hole_base, quat_peg_tip).copy()            
            hole_top_obs = pos[2] - pos[0]
        else:
            pos_obs = pos[1].copy()
            rot_obs = mat2Quat(rot[1])
            hole_top_obs = pos[2]

        peg_tip_lin_vel = jacp.dot(self.sim.data.qvel)
        peg_tip_rot_vel = jacr.dot(self.sim.data.qvel)

        if self.in_peg_frame:
            pos_obs = rot[0].T.dot(pos_obs)
            hole_top_obs = rot[0].T.dot(hole_top_obs)
            peg_tip_lin_vel = rot[0].T.dot(peg_tip_lin_vel)
            peg_tip_rot_vel = rot[0].T.dot(peg_tip_rot_vel)
        
        return np.concatenate([pos_obs, rot_obs, peg_tip_lin_vel, peg_tip_rot_vel, hole_top_obs])

    def _reset_state(self):
        '''
        Reset the robot state and return the observation.
        '''

        qvel = np.zeros(7)

        if self.sample_good_states and self.np_random.uniform() < 0.5:
            qpos = self.np_random.choice(self.good_states)
            self.set_state(qpos, qvel)
            self.sim.forward()
        else:
            qpos = self.good_states[-1] + self.np_random.uniform(-self.init_randomness,self.init_randomness,7)
            self.set_state(qpos, qvel)
            self.sim.forward()
            while self.sim.data.ncon > 0:
                qpos = self.good_states[-1] + self.np_random.uniform(-self.init_randomness,self.init_randomness,7)
                self.set_state(qpos, qvel)
                self.sim.forward()

    def _reset_target(self):
        '''
        Resets the hole position
        '''
        hole_data = self.np_random.choice(self.reachable_holes)
        self.good_states = hole_data['good_poses']
        self.sim.data.set_mocap_pos('hole', hole_data['hole_pos'])
        self.sim.data.set_mocap_quat('hole', hole_data['hole_quat'])

# [-0.00336991,  0.76797655, -0.01091426, -1.11394698,  0.0388207 ,  1.3678605 , -0.0077345 ]
# [-1.03213825e-08,  6.68086145e-01,  1.85550710e-08, -1.12884164e+00,  -1.17948636e-08,  1.34466485e+00,  6.88901499e-09]