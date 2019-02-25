import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from gym_kuka_mujoco.envs import KukaEnv
from gym_kuka_mujoco.utils.kinematics import forwardKinSite
from gym_kuka_mujoco.utils.quaternion import mat2Quat, subQuat


def compute_distance_travelled(x_pos, x_rot):
    pos_dist = 0
    rot_dist = 0
    for i in range(len(x_pos)-1):
        dx = x_pos[i+1] - x_pos[i]
        pos_dist += np.linalg.norm(dx)

        x_quat = mat2Quat(x_rot[i])
        x_quat_next = mat2Quat(x_rot[i+1])
        dq = subQuat(x_quat, x_quat_next)
        rot_dist += np.linalg.norm(dq)

    return pos_dist, rot_dist


def compute_average_ee_travel(controller, controller_options):
    env_options = {
        "model_path": "full_kuka_mesh_collision.xml",
        "frame_skip": 50,
        "time_limit": 2.0
    }
    # Create the environment
    orig_env = KukaEnv(controller=controller,
                  controller_options=controller_options,
                  **env_options)
    env = DummyVecEnv([lambda: orig_env])
    model = PPO2('MlpPolicy', env)

    # Simulate and get the average path length
    pos_dist_list = []
    rot_dist_list = []
    N_trials = 100
    for i in range(N_trials):
        x_pos = []
        x_rot = []
        obs = orig_env.reset()
        pos, rot = forwardKinSite(orig_env.sim, "ee_site", recompute=True)
        x_pos.append(pos.copy())
        x_rot.append(rot.copy())

        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=False)
            obs, rew, done, info = orig_env.step(act, render=False)
            pos, rot = forwardKinSite(orig_env.sim, "ee_site", recompute=True)
            x_pos.append(pos.copy())
            x_rot.append(rot.copy())
        pos_dist, rot_dist = compute_distance_travelled(x_pos, x_rot)
        pos_dist_list.append(pos_dist)
        rot_dist_list.append(rot_dist)

    return np.mean(pos_dist_list), np.mean(rot_dist_list), np.std(pos_dist_list), np.std(rot_dist_list)

if __name__ == "__main__":
    controller = "RelativeInverseDynamicsController"
    controller_options = {
        "model_path": "full_kuka_no_collision.xml"
    }
    
    pos_mean, rot_mean, pos_std, rot_std = compute_average_ee_travel(controller, controller_options)

    print("distance travelled mean: {}".format(pos_mean))
    print("distance travelled std:  {}".format(pos_std))
    print("rotation travelled mean: {}".format(rot_mean))
    print("rotation travelled std:  {}".format(rot_std))
    
    controller = "ImpedanceControllerV2"
    controller_options = {
        "model_path": "full_kuka_no_collision.xml",
        "pos_scale": 1.0,
        "rot_scale": 0.3,
        "stiffness": [1., 1., 1., 3., 3., 3.],
    }

    pos_mean, rot_mean, pos_std, rot_std = compute_average_ee_travel(controller, controller_options)

    print("distance travelled mean: {}".format(pos_mean))
    print("distance travelled std:  {}".format(pos_std))
    print("rotation travelled mean: {}".format(rot_mean))
    print("rotation travelled std:  {}".format(rot_std))
