import numpy as np

from gym_kuka_mujoco.envs import id_controlled_kuka_env
from gym_kuka_mujoco.utils.kinematics import forwardKin

class PegInsertionEnv(id_controlled_kuka_env.DiffIdControlledKukaEnv):
    setpoint_diff = True
    def __init__(self, *args, **kwargs):
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

    def get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        peg_tip_pos, _ = forwardKin(self.sim, self.peg_tip_pos, np.array([1.,0.,0.,0.]), self.peg_body_id)
        err = self.hole_base_pos - peg_tip_pos
        if self.use_shaped_reward:
            # quadratic cost on the error and action
            reward = -err.dot(self.Q).dot(err) - action.dot(self.R).dot(action)
            reward += 1.0 if err.dot(err) < self.eps else 0.0
            return reward
        else:
            # sparse reward
            return 1.0 if err.dot(err) < self.eps else 0.0