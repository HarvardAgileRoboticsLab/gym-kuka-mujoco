import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py.builder import MujocoException

class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    use_shaped_reward = True
    def __init__(self, model_path=None):
        '''
        Constructs the file, sets the time limit and calls the constructor of
        the super class.
        '''
        utils.EzPickle.__init__(self)
        if model_path is None:
            model_path = 'full_kuka_no_collision.xml'
            model_path = 'full_kuka_no_collision_no_gravity.xml'
        full_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'assets', model_path)
        
        self.time_limit = 3

        # Parameters for the cost function
        self.state_des = np.zeros(14)
        self.Q = np.eye(14)
        self.Q = np.diag([1,1,1,1,1,1,1,.01,.01,.01,.01,.01,.01,.01])
        self.R = 1e-2*np.eye(7)
        self.eps = 1e-1

        # Call the super class
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, full_path, 20)
        self.initialized = True

        self.torque_scaling = self.subtree_mass()
        self.torque_scaling /= np.max(self.torque_scaling)
        self.torque_scaling*=10.
        # self.torque_scaling*=2.
        # self.torque_scaling*=.1
        low = self.action_space.low/self.torque_scaling
        high = self.action_space.high/self.torque_scaling
        self.action_space = spaces.Box(low, high, dtype=self.action_space.low.dtype)

    def subtree_mass(self):
        '''
        Compute the subtree mass of the Kuka Arm using the actual link names.
        '''
        body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
        body_ids = [self.model.body_name2id(n) for n in body_names]
        return self.model.body_subtreemass[body_ids]

    def update_action(self, a):
        '''
        This function is called once per step.
        '''
        self.action = a*self.torque_scaling

    def get_torque(self):
        '''
        This function is called multiple times per step to simulate a
        low-level controller.
        '''
        return self.action

    def get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        err = self.state_des - state
        if self.use_shaped_reward:
            # quadratic cost on the error and action
            reward = -err.dot(self.Q).dot(err) - action.dot(self.R).dot(action)
            # reward += -np.sqrt(err.dot(self.Q).dot(err))
            # reward += -np.log(err.dot(self.Q).dot(err) + 1e-3)
            # reward = -err.dot(self.Q).dot(err) - action.dot(self.R).dot(action)
            # reward += 1.0 if err.dot(err) < self.eps else 0.0
            return reward, {}
        else:
            # sparse reward
            return (1.0, {}) if err.dot(err) < self.eps else (0.0, {})    

    def step(self, action, render=False):
        '''
        Simulate for `self.frame_skip` timesteps. Calls update_action() once
        and then calls get_torque() repeatedly to simulate a low-level
        controller.
        Optional argument render will render the intermediate frames for a smooth animation.
        '''
        # Hack to return an observation during the super class __init__ method.
        if not self.initialized:
            return self._get_obs(), 0, False, {}

        # Set the action to be used for the simulation.
        self.update_action(action)

        # Get the reward from the state and action.
        state = np.concatenate((self.sim.data.qpos[:], self.sim.data.qvel[:]))

        # Simulate the low level controller.
        dt = self.sim.model.opt.timestep
        try:
            total_reward = 0
            total_reward_info = dict()
            for _ in range(self.frame_skip):
                self.sim.data.ctrl[:] = self.get_torque()
                self.sim.step()
                if not np.all(np.isfinite(self.sim.data.qpos)):
                    print("Warning: simulation step returned inf or nan.")
                reward, reward_info = self.get_reward(state, action)
                total_reward += reward*dt
                for k, v in reward_info.items():
                    if 'reward' in k:
                        total_reward_info[k] = total_reward_info.get(k,0) + v*self.sim.model.opt.timestep
                if render:
                    self.render()

            # Get observation and check finished
            done = (self.sim.data.time > self.time_limit) or self.get_done()
            obs = self._get_obs()
        except MujocoException as e:
            print(e)
            reward = 0
            obs = np.zeros_like(self.action_space.low)
            done = True

        return obs, total_reward, done, total_reward_info

    def get_done(self):
        return False

    def viewer_setup(self):
        '''
        Make the viewer point at the base.
        '''
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        '''
        Reset the robot state and return the observation.
        '''
        while(True):
            try:
                qpos = 0.1*self.np_random.uniform(low=self.model.jnt_range[:,0], high=self.model.jnt_range[:,1], size=self.model.nq)
                # qpos = self.np_random.uniform(low=self.model.jnt_range[:,0], high=self.model.jnt_range[:,1], size=self.model.nq)
                # qpos = 0.01*self.np_random.uniform(low=self.model.jnt_range[:,0], high=self.model.jnt_range[:,1], size=self.model.nq)
                qvel = np.zeros(7)
    
                self.set_state(qpos, qvel)
            except MujocoException as e:
                print(e)
                continue
            break
        return self._get_obs()

    def _get_obs(self):
        '''
        Return the full state as the observation
        '''
        if not self.initialized:
            return np.zeros(14)

        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
