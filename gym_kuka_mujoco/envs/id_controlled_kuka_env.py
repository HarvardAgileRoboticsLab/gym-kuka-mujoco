import numpy as np
import mujoco_py
from gym import spaces
from gym_kuka_mujoco.envs import kuka_env

class IdControlledKukaEnv(kuka_env.KukaEnv):
    def __init__(self,
                 kp_id=None,
                 kd_id=None,
                 kp_pd=None,
                 kd_pd=None,
                 **kwargs):
        super(IdControlledKukaEnv, self).__init__(**kwargs)

        # Set the action space
        low_pos = self.model.jnt_range[:,0]
        high_pos = self.model.jnt_range[:,1]

        low_vel = -3*np.ones(self.model.nv)
        high_vel = 3*np.ones(self.model.nv)

        low = np.concatenate((low_pos, low_vel))
        high = np.concatenate((high_pos, high_vel))

        self.action_space = spaces.Box(high, low, dtype=np.float32)

        # Set the controller gains
        self.kp_id = kp_id if kp_id is not None else 100
        self.kd_id = kd_id if kd_id is not None else 2 * np.sqrt(self.kp_id)
        self.kp_pd = kp_pd if kp_pd is not None else \
                    1e-2*self.model.body_subtreemass[2:]
        self.kd_pd = kd_pd if kd_pd is not None else \
                    np.minimum(self.kp_pd*self.model.opt.timestep,
                        2*np.sqrt(self.model.body_subtreemass[2:]*self.kp_pd))

    def step(self, a):
        # Hack to allow different sized action space than the super class
        if len(a) == self.model.nu:
            return super(IdControlledKukaEnv, self).step(a)

        # Get the position and velocity
        qpos_set = a[:7]
        qvel_set = a[7:14]

        # Compute position and velocity errors
        qpos_err = qpos_set - self.sim.data.qpos
        qvel_err = qvel_set - self.sim.data.qvel

        # Compute desired acceleration using inner loop PD law
        self.sim.data.qacc[:] = self.kp_id*qpos_err + self.kd_id*qvel_err
        mujoco_py.functions.mj_inverse(self.model,self.sim.data)
        id_torque = self.sim.data.qfrc_inverse[:]/300

        # Compute torque from outer loop PD law
        pd_torque = self.kp_pd*qpos_err + self.kd_pd*qvel_err

        # Sum the torques
        total_torque = id_torque + pd_torque

        return super(IdControlledKukaEnv, self).step(total_torque)
