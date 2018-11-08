import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    use_shaped_reward = True
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

        # Parameters for the cost function
        self.state_des = np.zeros(14)
        self.Q = 1e-2*np.eye(14)
        self.R = 1e-6*np.eye(7)
        self.eps = 1e-1

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

    def step(self, action):
        '''
        Simulate for `self.frame_skip` timesteps. Calls update_action() once
        and then calls get_torque() repeatedly to simulate a low-level
        controller.
        '''
        # Set the action to be used for the simulation.
        self.update_action(action)

        # Get the reward from the state and action.
        state = np.concatenate((self.sim.data.qpos[:], self.sim.data.qvel[:]))
        reward = self.get_reward(state, action)

        # Simulate the low level controller.
        for _ in range(self.frame_skip):
            self.sim.data.ctrl[:] = self.get_torque()
            self.sim.step()

        # Get observation and check finished
        done = self.sim.data.time > self.time_limit
        obs = self._get_obs()
        return obs, reward, done, {}

    def viewer_setup(self):
        '''
        Make the viewer point at the base.
        '''
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        '''
        Reset the robot state and return the observation.
        '''
        if np.random.random() > -0.5:
                qpos = 0.1*self.np_random.uniform(low=self.model.jnt_range[:,0], high=self.model.jnt_range[:,1], size=self.model.nq)
                qvel = np.zeros(7)
        else:
            qpos = np.zeros(7)
            qvel = np.zeros(7)
        
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        '''
        Return the full state as the observation
        '''
        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
