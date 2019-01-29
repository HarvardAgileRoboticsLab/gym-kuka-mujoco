import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py.builder import MujocoException

class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    random_model = False
    random_target = False

    info_keywords = ('distance',)
    
    def __init__(self, model_path=None):
        '''
        Constructs the file, sets the time limit and calls the constructor of
        the super class.
        '''
        utils.EzPickle.__init__(self)
        if model_path is None:
            model_path = 'full_kuka_no_collision_no_gravity.xml'
        full_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'assets', model_path)
        
        self.time_limit = 3

        # Parameters for the cost function
        self.state_des = np.zeros(14)
        self.Q = np.diag([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

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

    def viewer_setup(self):
        '''
        Overwrites the MujocoEnv method to make the camera point at the base.
        '''
        self.viewer.cam.trackbodyid = 0

    '''
    Step and helper methods. Only overwrite the helpers in the subclasses.
    '''
    def step(self, action, render=False):
        '''
        Simulate for `self.frame_skip` timesteps. Calls _update_action() once
        and then calls _get_torque() repeatedly to simulate a low-level
        controller.
        Optional argument render will render the intermediate frames for a smooth animation.
        '''
        # Hack to return an observation during the super class __init__ method.
        if not self.initialized:
            return self._get_obs(), 0, False, {}

        # Set the action to be used for the simulation.
        self._update_action(action)

        # Get the reward from the state and action.
        state = np.concatenate((self.sim.data.qpos[:], self.sim.data.qvel[:]))

        # Simulate the low level controller.
        dt = self.sim.model.opt.timestep

        try:
            total_reward = 0
            total_reward_info = dict()
            for _ in range(self.frame_skip):
                self.sim.data.ctrl[:] = self._get_torque()
                self.sim.step()
                if not np.all(np.isfinite(self.sim.data.qpos)):
                    print("Warning: simulation step returned inf or nan.")
                reward, reward_info = self._get_reward(state, action)
                total_reward += reward*dt
                for k, v in reward_info.items():
                    if 'reward' in k:
                        total_reward_info[k] = total_reward_info.get(k,0) + v*self.sim.model.opt.timestep
                if render:
                    self.render()

            # Get observation and check finished
            done = (self.sim.data.time > self.time_limit) or self._get_done()
            obs = self._get_obs()
            info = self._get_info()
            info.update(total_reward_info)
        except MujocoException as e:
            print(e)
            reward = 0
            obs = np.zeros_like(self.observation_space.low)
            done = True
            info = {}

        return obs, total_reward, done, info

    def _update_action(self, a):
        '''
        This function is called once per step.
        '''
        self.action = a*self.torque_scaling

    def _get_torque(self):
        '''
        This function is called multiple times per step to simulate a
        low-level controller.
        '''
        return self.action

    def _get_obs(self):
        '''
        Return the full state as the observation
        '''

        if self.random_target:
            return np.concatenate((self._get_state_obs(), self._get_target_obs()))
        else:
            return self._get_state_obs()

    def _get_state_obs(self):
        '''
        Return the observation given by the state.
        '''
        if not self.initialized:
            return np.zeros(14)

        return np.concatenate([self.sim.data.qpos[:], self.sim.data.qvel[:]])

    def _get_target_obs(self):
        '''
        Return the observation given by the goal for the episode.
        '''
        return self.state_des[:7]

        
    def _get_reward(self, state, action):
        '''
        Compute single step reward.
        '''
        err = self.state_des - state
        # quadratic cost on the state error
        reward = -err.dot(self.Q).dot(err)
        return reward, {}

    def _get_done(self):
        '''
        Check the termination condition.
        '''
        return False

    def _get_info(self):
        '''
        Get any additional info.
        '''
        q_err = self.state_des[:7] - self.sim.data.qpos
        v_err = self.state_des[7:] - self.sim.data.qvel
        dist = q_err.dot(q_err)
        velocity = v_err.dot(v_err)
        return {
            'distance': dist,
            'velocity': velocity
        }

    '''
    Reset and helper methods. Only overwrite the helper methods in subclasses.
    '''
    def reset_model(self):
        '''
        Overwrites the MujocoEnv method to reset the robot state and return the observation.
        '''
        while(True):
            try:
                self._reset_state()
                if self.random_model:
                    self._reset_model_params()
                if self.random_target:
                    self._reset_target()
            except MujocoException as e:
                print(e)
                continue
            break

        return self._get_obs()

    def _reset_state(self):
        '''
        Reset the state of the model (i.e. the joint positions and velocities).
        '''
        qpos = 0.1*self.np_random.uniform(low=self.model.jnt_range[:,0], high=self.model.jnt_range[:,1], size=self.model.nq)
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)

    def _reset_target(self):
        '''
        Reset the goal parameters. Target pose for the base environment, but may change with subclasses.
        '''
        self.state_des[:7] = self.np_random.uniform(self.model.jnt_range[:,0], self.model.jnt_range[:,1])

    def _reset_model_params(self):
        '''
        TODO: implement this for domain randomization.
        '''
        raise NotImplementedError


# This class is a hack to get around a bad action space initialized with the SAC policy
class KukaEnvSAC(KukaEnv):
    def __init__(self, *args, **kwargs):
        super(KukaEnvSAC, self).__init__(*args, **kwargs)
        self.action_space = spaces.Box(-10*np.ones(7), 10*np.ones(7))
