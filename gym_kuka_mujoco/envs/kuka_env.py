import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        '''
        Constructs the file, sets the time limit and calls the constructor of
        the super class.
        '''
        utils.EzPickle.__init__(self)
        model_path = 'kuka_model_no_collision.xml'
        full_path = os.path.join(
            os.path.dirname(__file__), 'assets', model_path)
        self.time_limit = 3
        mujoco_env.MujocoEnv.__init__(self, full_path, 2)

    def update_action(self, a):
        '''
        This function is called once per step.
        '''
        self.action = a

    def get_torque(self):
        '''
        This function is called multiple times per step to simulate a
        low-level controller.
        '''
        return self.action/300.0

    def step(self, a):
        '''
        Simulate for `self.frame_skip` timesteps. Calls update_action() once
        and then calls get_torque() repeatedly to simulate a low-level
        controller.
        '''
        # Get torque from action
        self.update_action(a)

        # Get the reward
        reward_dist = -np.linalg.norm(self.sim.data.qpos)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # Simulate the low level controller
        for _ in range(self.frame_skip):
            self.sim.data.ctrl[:] = self.get_torque()
            self.sim.step()

        # Get observation and check finished
        done = self.sim.data.time > self.time_limit
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        '''
        Make the viewer point at the base.
        '''
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        '''
        Reset the robot state and return the observation.
        '''
        qpos = self.init_qpos + self.np_random.uniform(
                                    low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
                                    low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        '''
        Return the full state as the observation
        '''
        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
