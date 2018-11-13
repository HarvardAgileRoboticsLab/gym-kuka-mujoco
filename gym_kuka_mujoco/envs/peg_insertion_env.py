from gym_kuka_mujoco.envs import id_controlled_kuka_env

class PegInsertionEnv(id_controlled_kuka_env.DiffIdControlledKukaEnv):
    setpoint_diff = True
    def __init__(self, *args, **kwargs):
        kwargs['model_path'] = kwargs.get('model_path', 'full_peg_insertion_experiment.xml')
        super(PegInsertionEnv, self).__init__(*args, **kwargs)
        self.time_limit = 10

    def get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        err = self.state_des - state
        if self.use_shaped_reward:
            # quadratic cost on the error and action
            reward = -err.dot(self.Q).dot(err) - action.dot(self.R).dot(action)
            reward += 1.0 if err.dot(err) < self.eps else 0.0
            return reward
        else:
            # sparse reward
            return 1.0 if err.dot(err) < self.eps else 0.0