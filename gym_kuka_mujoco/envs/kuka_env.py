import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class KukaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = 'kuka_model_no_collision.xml'
        full_path = os.path.join(
            os.path.dirname(__file__), 'assets', model_path)
        self.time_limit = 3
        mujoco_env.MujocoEnv.__init__(self, full_path, 2)

    def step(self, a):
        reward_dist = -np.linalg.norm(self.sim.data.qpos)
        a = a * 0.01
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        done = self.sim.data.time > self.time_limit
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
