import os
import random
import numpy as np

from gym_kuka_mujoco.envs import kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat, quatAdd
from gym_kuka_mujoco.envs.assets import kuka_asset_dir

class HoldPositionEnv(kuka_env.KukaEnv):
    '''
    An environment designed to train a policy to be tested on hardware.
    '''
    
    def __init__(self,
                 *args,
                 init_randomness=0.00,
                 joint_force_randomness=0.00,
                 init_qpos=None,
                 target_qpos=None,
                 use_qpos_cost=False,
                 use_qvel_cost=False,
                 use_pos_cost=True,
                 use_rot_cost=True,
                 qpos_cost=1.,
                 qvel_cost=1.,
                 pos_cost=1.,
                 rot_cost=1.,
                 use_joint_observation=True,
                 use_rel_pose_observation=True,
                 **kwargs):
        
        # Store options
        self.use_qpos_cost = use_qpos_cost
        self.use_qvel_cost = use_qvel_cost
        self.use_pos_cost = use_pos_cost
        self.use_rot_cost = use_rot_cost

        self.init_randomness = np.ones(7)*init_randomness
        self.joint_force_randomness = np.ones(7)*joint_force_randomness
        self.use_rel_pose_observation = use_rel_pose_observation
        self.use_joint_observation = use_joint_observation
        
        if target_qpos is None:
            self.target_qpos = np.array([np.pi/2,
                                        -np.pi/6,
                                        -np.pi/3,
                                        -np.pi/2,
                                         np.pi*3/4,
                                        -np.pi/4,
                                         0.0])
        else:
            self.target_qpos = target_qpos.copy()

        if init_qpos is None:
            self._init_qpos = self.target_qpos.copy()
        else:
            if isinstance(init_qpos, np.ndarray):
                self._init_qpos = init_qpos.copy()
            else:
                self._init_qpos = np.array(init_qpos)
        
        # Set the default model path.
        kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole_no_gravity.xml')       
        super(HoldPositionEnv, self).__init__(*args, **kwargs)
        
        # Compute the target pos and quat from forwardKinSite.
        self.set_state(self.target_qpos, np.zeros(7))
        self.sim.forward()
        pos, rot = forwardKinSite(self.sim, ['peg_tip'], recompute=True)
        self.target_pos = pos[0].copy()
        self.target_quat = mat2Quat(rot[0])

        # Keep the cost terms
        if self.use_rot_cost:
            self.Q_qpos = np.eye(7)*qpos_cost
        if self.use_qvel_cost:
            self.Q_qvel = np.eye(7)*qvel_cost
        if self.use_pos_cost:
            self.Q_pos = np.eye(3)*pos_cost
        if self.use_rot_cost:
            self.Q_rot = np.eye(3)*rot_cost

    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''

        # Compute peg tip velocity.        
        # peg_tip_id = self.model.site_name2id('peg_tip')
        # jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id, recompute=False)
        # peg_tip_vel = jacp.dot(self.data.qvel[:])

        reward_info = dict()
        reward = 0.

        if self.use_qpos_cost:
            qpos_err = self.sim.data.qpos - self.target_qpos
            reward_info['qpos_reward'] = -qpos_err.dot(self.Q_qpos).dot(qpos_err)
            reward += reward_info['qpos_reward']
        if self.use_qvel_cost:
            reward_info['qvel_reward'] = -self.data.qvel.dot(self.Q_qvel).dot(self.data.qvel)
            reward += reward_info['qvel_reward']
        if self.use_pos_cost or self.use_rot_cost:
            # compute position and rotation error
            pos, rot = forwardKinSite(self.sim, ['peg_tip'], recompute=False)
            peg_pos = pos[0]
            peg_quat = mat2Quat(rot[0])
        if self.use_pos_cost:
            pos_err = peg_pos - self.target_pos
            reward_info['pos_reward'] = -pos_err.dot(self.Q_pos).dot(pos_err)
            reward += reward_info['pos_reward']
        if self.use_rot_cost:
            rot_err = subQuat(peg_quat, self.target_quat)
            reward_info['rot_reward'] = -rot_err.dot(self.Q_rot).dot(rot_err)
            reward += reward_info['rot_reward']

        return reward, reward_info

    def _get_info(self):
        # Build the info dict.
        info = dict()

        # Joint space distances.
        qpos_err = self.data.qpos - self.target_qpos
        info['qpos_dist'] = np.linalg.norm(qpos_err)
        info['qvel_dist'] = np.linalg.norm(self.data.qvel)

        # Task space distances.
        pos, rot = forwardKinSite(self.sim, ['peg_tip'], recompute=False)
        pos_err = pos[0] - self.target_pos
        info['pos_dist'] = np.linalg.norm(pos_err)
        rot_err = subQuat(mat2Quat(rot[0]), self.target_quat)
        info['rot_dist'] = np.linalg.norm(rot_err)

        # Success metric.
        info['success'] = float(info['pos_dist'] < 1e-2)
        return info

    def _get_state_obs(self):
        '''
        Compute the observation at the current state.
        '''

        # Return superclass observation.
        if self.use_joint_observation:
            obs = super(HoldPositionEnv, self)._get_state_obs()
        else:
            obs = np.zeros(0)

        # Compute the relative pose information.
        if self.use_rel_pose_observation:
            if self.initialized:
                pos, rot = forwardKinSite(self.sim, ['peg_tip'], recompute=False)
                pos_err = pos[0] - self.target_pos
                rot_err = subQuat(mat2Quat(rot[0]), self.target_quat)
            else:
                pos_err = np.zeros(3)
                rot_err = np.zeros(3)
            obs = np.concatenate([obs, pos_err, rot_err])

        return obs

    def _get_random_applied_torques(self):
        return self.np_random.uniform(-self.joint_force_randomness, self.joint_force_randomness)

    def _reset_state(self):
        '''
        Reset the simulation state.
        '''
        qpos = self._init_qpos.copy()
        qpos += self.np_random.uniform(-self.init_randomness, self.init_randomness)
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)
        self.sim.forward()