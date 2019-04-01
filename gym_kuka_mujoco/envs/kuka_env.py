import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py.builder import MujocoException

from .assets import kuka_asset_dir
from gym_kuka_mujoco.controllers import controller_registry

class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    default_info = dict()
    def __init__(self,
                 controller,
                 controller_options,
                 model_path='full_kuka_no_collision_no_gravity.xml',
                 frame_skip=20,
                 time_limit=3.,
                 timestep=0.002,
                 random_model = False,
                 random_target = False,
                 quadratic_pos_cost = True,
                 quadratic_vel_cost = False):
        '''
        Constructs the file, sets the time limit and calls the constructor of
        the super class.
        '''
        self.random_model = random_model
        self.random_target = random_target
        self.quadratic_pos_cost = quadratic_pos_cost
        self.quadratic_vel_cost = quadratic_vel_cost

        utils.EzPickle.__init__(self)

        full_path = os.path.join(kuka_asset_dir(), model_path)
        
        self.time_limit = time_limit

        # Parameters for the cost function
        self.state_des = np.zeros(14)
        self.Q_pos = np.diag([1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.])
        self.Q_vel = np.diag([0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.])

        # Call the super class
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, full_path, frame_skip)
        self.model.opt.timestep = timestep
        self.initialized = True

        # Create the desired controller.
        controller_cls = controller_registry[controller]
        self.controller = controller_cls(sim=self.sim, **controller_options)
        
        # Take the action space from the controller.
        self.action_space = self.controller.action_space

        self.last_action = None

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
        self.last_action = action

        # Get the reward from the state and action.
        state = np.concatenate((self.sim.data.qpos[:], self.sim.data.qvel[:]))

        # Simulate the low level controller.
        dt = self.sim.model.opt.timestep

        try:
            total_reward = 0
            total_reward_info = dict()
            for _ in range(self.frame_skip):
                torque = self._get_torque()
                self.sim.data.ctrl[:] = np.clip(torque, -300, 300)
                self.sim.data.qfrc_applied[:] = self._get_random_applied_force()
                self.sim.step()
                if not np.all(np.isfinite(self.sim.data.qpos)):
                    print("Warning: simulation step returned inf or nan.")
                reward, reward_info = self._get_reward(state, action)
                total_reward += reward*dt
                for k, v in reward_info.items():
                    if 'reward' in k:
                        total_reward_info[k] = total_reward_info.get(k,0) + v*dt
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
            info = self.default_info

        return obs, total_reward, done, info

    def _update_action(self, a):
        '''
        This function is called once per step.
        '''
        self.controller.set_action(a)

    def _get_torque(self):
        '''
        This function is called multiple times per step to simulate a
        low-level controller.
        '''
        return self.controller.get_torque()

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
        # quadratic cost on the state error
        reward_info = dict()
        reward = 0.

        err = self.state_des - state
        if self.quadratic_pos_cost:
            reward_info['quadratic_pos_cost'] = -err.dot(self.Q_pos).dot(err)
            reward += reward_info['quadratic_pos_cost']

        if self.quadratic_vel_cost:
            reward_info['quadratic_vel_cost'] = -err.dot(self.Q_vel).dot(err)
            reward += reward_info['quadratic_vel_cost']

        return reward, reward_info

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
        dist = np.sqrt(q_err.dot(q_err))
        velocity = np.sqrt(v_err.dot(v_err))
        return {
            'distance': dist,
            'velocity': velocity
        }

    def _get_random_applied_force(self):
        return np.zeros(self.model.nv)

    '''
    Reset and helper methods. Only overwrite the helper methods in subclasses.
    '''
    def reset_model(self):
        '''
        Overwrites the MujocoEnv method to reset the robot state and return the observation.
        '''
        while(True):
            try:
                if self.random_model:
                    self._reset_model_params()
                if self.random_target:
                    self._reset_target()
                self._reset_state() # Always reset the state after the target is reset.
                self.sim.forward()
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