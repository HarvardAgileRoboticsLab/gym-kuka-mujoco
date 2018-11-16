import random
import numpy as np

from gym_kuka_mujoco.envs import id_controlled_kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples

class PegInsertionEnv(id_controlled_kuka_env.DiffIdControlledKukaEnv):
    setpoint_diff = True
    sample_good_states = True
    use_ft_sensor = True
    hole_size = "tight"
    
    def __init__(self, *args, **kwargs):
        if self.hole_size == "big":
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_big_hole.xml')
        elif self.hole_size == "mid":
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_mid_hole.xml')
        elif self.hole_size == "small":
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment_small_hole.xml')
        else:
            kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment.xml')
        super(PegInsertionEnv, self).__init__(*args, **kwargs)

        self.time_limit = 10
        self.peg_body_id = self.model.body_name2id('peg')
        self.peg_tip_pos = np.array([0,0,0.10902])

        hole_body_id = self.model.body_name2id('hole')
        hole_local_pos = np.array([0., 0., 0.])
        hole_local_quat = np.array([1., 0., 0., 0.])
        self.hole_base_pos, _ = forwardKin(self.sim, hole_local_pos, hole_local_quat, hole_body_id)
        self.Q = np.eye(3)
        self.eps = 1e-2

        if self.sample_good_states:
            self.good_states = hole_insertion_samples(self.sim)

    def get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        peg_tip_pos, _ = forwardKin(self.sim, self.peg_tip_pos, np.array([1.,0.,0.,0.]), self.peg_body_id)
        err = self.hole_base_pos - peg_tip_pos
        dist = np.sqrt(err.dot(err))
        if self.use_shaped_reward:
            # quadratic cost on the error and action
            reward = -err.dot(self.Q).dot(err) - action.dot(self.R).dot(action)
            reward += 1.0 if dist < self.eps else 0.0
            return reward
        else:
            # sparse reward
            return 1.0 if dist < self.eps else 0.0

    def get_done(self):
        peg_tip_pos, _ = forwardKin(self.sim, self.peg_tip_pos, np.array([1.,0.,0.,0.]), self.peg_body_id)
        err = self.hole_base_pos - peg_tip_pos
        dist = np.sqrt(err.dot(err))
        return dist < self.eps


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
        if self.sample_good_states and np.random.random() < .5:
            qpos = random.choice(self.good_states)
        else:
            qpos = np.array([0., 0.94719755, 0., -1.49719755, 0., 0.69719755, 0.])
        
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)

        return self._get_obs()

class PegInsertionBigHoleEnv(PegInsertionEnv):
    hole_size = "big"

class PegInsertionMidHoleEnv(PegInsertionEnv):
    hole_size = "mid"

class PegInsertionSmallHoleEnv(PegInsertionEnv):
    hole_size = "small"
