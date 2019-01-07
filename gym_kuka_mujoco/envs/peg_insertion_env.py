import random
import numpy as np

from gym_kuka_mujoco.envs import id_controlled_kuka_env
from gym_kuka_mujoco.envs import remote_center_controlled_kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinSite, forwardKinJacobianSite
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
from gym_kuka_mujoco.utils.projection import rotate_cost_by_matrix

class PegInsertionEnv(id_controlled_kuka_env.DiffIdControlledKukaEnv):
    setpoint_diff = True
    sample_good_states = True
    use_ft_sensor = True
    # hole_size = "tight"
    
    def __init__(self, *args, hole_id=None, **kwargs):
        print("PegInsertionEnv __init__()")
        if hole_id is None:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment.xml')
        elif hole_id >= 0:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_moving_hole_id={:03d}.xml'.format(hole_id))
        else:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_no_hole.xml')       
        kwargs['control_model_path'] = kwargs.get('control_model_path', 'full_peg_insertion_experiment_no_collision.xml')
        super(PegInsertionEnv, self).__init__(*args, **kwargs)

        self.time_limit = 3
        self.Q_pos = np.diag([100,100,10])
        self.Q_vel = np.diag([1,1,.1])
        self.eps = 1e-2

        self.frame_skip=10

        if self.sample_good_states:
            self.good_states = hole_insertion_samples(self.sim, range=[0.,0.06])

    def get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        # peg_tip_pos, _ = forwardKin(self.sim, self.peg_tip_pos, np.array([1.,0.,0.,0.]), self.peg_body_id)
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        err = pos[0] - pos[1]
        dist = np.sqrt(err.dot(err))
        peg_tip_id = self.model.site_name2id('peg_tip')
        jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id)
        peg_tip_vel = jacp.dot(self.data.qvel[:])
        # print("reward_dist: {}".format(dist))
        reward_info = dict()
        if self.use_shaped_reward:
            # quadratic cost on the error and action
            # rotate the cost terms to align with the hole
            Q_pos = rotate_cost_by_matrix(self.Q_pos,rot[1].T)
            Q_vel = rotate_cost_by_matrix(self.Q_vel,rot[1].T)

            reward_info['position_reward'] = -np.sqrt(err.dot(Q_pos).dot(err))
            reward_info['velocity_reward'] = -np.sqrt(peg_tip_vel.dot(Q_vel).dot(peg_tip_vel)) 
            reward = reward_info['position_reward']
            reward += reward_info['velocity_reward']
            #reward -= action.dot(self.R).dot(action)
            # reward += 10.0 if dist < self.eps else 0.0
            # reward -= 100.0 if dist > .2 else 0.0
        else:
            # sparse reward
            reward_info['sparse_reward'] = 1.0 if dist < self.eps else 0.0
            reward = reward_info['sparse_reward']
        
        return reward, reward_info

    def get_done(self):
        pos, _ = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        err = pos[0] - pos[1]
        dist = np.sqrt(err.dot(err))
        # print("done_dist:   {}".format(dist))
        # if dist < self.eps:
            # print("done")
        #return dist < self.eps or dist > .2 # terminate if > 20cm
        return False

    def _get_obs(self):
        '''
        Compute the observation at the current state.
        '''

        # Return superclass observation.
        obs = super(PegInsertionEnv, self)._get_obs()
        if not self.use_ft_sensor:
            return obs    

        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
        else:
            ft_obs = self.sim.data.sensordata

        obs = np.concatenate([obs, ft_obs])
        return obs

    def reset_model(self):
        '''
        Reset the robot state and return the observation.
        '''
        if self.sample_good_states and np.random.random() < 0*0.5:
            qpos = random.choice(self.good_states)
        else:
            qpos = self.good_states[-1]
            qpos += np.random.uniform(-.01,.01,7)
        
        # qpos = np.zeros(7)
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)

        return self._get_obs()

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
        self.Q_vel = np.diag([1,1,.1])
        self.eps = 1e-2

        self.frame_skip=10

        if self.sample_good_states:
            self.good_states = hole_insertion_samples(self.sim, range=[0.,0.06])

    def get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        # peg_tip_pos, _ = forwardKin(self.sim, self.peg_tip_pos, np.array([1.,0.,0.,0.]), self.peg_body_id)
        pos, rot = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        err = pos[0] - pos[1]
        dist = np.sqrt(err.dot(err))
        peg_tip_id = self.model.site_name2id('peg_tip')
        jacp, jacv = forwardKinJacobianSite(self.sim, peg_tip_id)
        peg_tip_vel = jacp.dot(self.data.qvel[:])
        # print("reward_dist: {}".format(dist))
        reward_info = dict()
        if self.use_shaped_reward:
            # quadratic cost on the error and action
            # rotate the cost terms to align with the hole
            Q_pos = rotate_cost_by_matrix(self.Q_pos,rot[1].T)
            Q_vel = rotate_cost_by_matrix(self.Q_vel,rot[1].T)

            reward_info['position_reward'] = -np.sqrt(err.dot(Q_pos).dot(err))
            reward_info['velocity_reward'] = -np.sqrt(peg_tip_vel.dot(Q_vel).dot(peg_tip_vel)) 
            reward = reward_info['position_reward']
            reward += reward_info['velocity_reward']
            #reward -= action.dot(self.R).dot(action)
            # reward += 10.0 if dist < self.eps else 0.0
            # reward -= 100.0 if dist > .2 else 0.0
        else:
            # sparse reward
            reward_info['sparse_reward'] = 1.0 if dist < self.eps else 0.0
            reward = reward_info['sparse_reward']
        
        return reward, reward_info

    def get_done(self):
        pos, _ = forwardKinSite(self.sim, ['peg_tip','hole_base'])
        err = pos[0] - pos[1]
        dist = np.sqrt(err.dot(err))
        # print("done_dist:   {}".format(dist))
        # if dist < self.eps:
            # print("done")
        #return dist < self.eps or dist > .2 # terminate if > 20cm
        return False

    def _get_obs(self):
        '''
        Compute the observation at the current state.
        '''

        # Return superclass observation.
        obs = super(RemoteCenterPegInsertionEnv, self)._get_obs()
        if not self.use_ft_sensor:
            return obs    

        # Return superclass observation stacked with the ft observation.
        if not self.initialized:
            ft_obs = np.zeros(6)
        else:
            ft_obs = self.sim.data.sensordata

        obs = np.concatenate([obs, ft_obs])
        return obs

    def reset_model(self):
        '''
        Reset the robot state and return the observation.
        '''
        if self.sample_good_states and np.random.random() < 0*0.5:
            qpos = random.choice(self.good_states)
        else:
            qpos = self.good_states[-1]
            qpos += np.random.uniform(-.01,.01,7)
        
        # qpos = np.zeros(7)
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)

        return self._get_obs()
