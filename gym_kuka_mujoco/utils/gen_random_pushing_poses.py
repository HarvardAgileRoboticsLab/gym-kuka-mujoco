from gym_kuka_mujoco.utils.kinematics import inverseKin
from gym_kuka_mujoco.utils.quaternion import identity_quat
from gym_kuka_mujoco.envs.assets import kuka_asset_dir
import os
import mujoco_py
import numpy as np

def gen_random_pushing_poses(sim, qpos_idx, N=10):

    # The points to be transformed.
    peg_body_id = sim.model.body_name2id('peg')

    # Compute the forward kinematics
    q_nom = np.zeros(7)
    q_init = np.random.random(7)
    peg_tip_idx = sim.model.site_name2id('peg_tip')
    body_pos = sim.model.site_pos[peg_tip_idx]
    
    # for 
    brick_pos = sim.data.qpos[7:10].copy()

    poses = []
    while (len(poses) < N):
        if N > 10:
            print("Generating pose {}".format(len(poses)+1))
        delta = np.random.normal(size=3)
        delta *= np.array([.075, 0.01, 0.])    
        # world_pos = np.array([0.7, 0., 1.22]) # above the block
        # world_pos = np.array([0.75, -0.04, 1.20]) # to the -y of the block
        if delta[1] > 0:
            delta[1] += 0.02
        else:
            delta[1] -= 0.02

        world_pos = brick_pos + delta
        
        angle = np.random.uniform(-np.pi, np.pi)
        world_quat = np.array([0., np.sin(angle/2), np.cos(angle/2), 0])

        try:
            q_opt = inverseKin(sim, q_init, q_nom, body_pos, world_pos, world_quat, peg_body_id, qpos_idx=qpos_idx, raise_on_fail=True)
        except RuntimeError as e:
            continue
        poses.append(q_opt)
    
    return poses



if __name__ == "__main__":

    model_filename = 'full_pushing_experiment_no_gravity.xml'
    model_path = os.path.join(kuka_asset_dir(), model_filename)

    # Load path
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    sim.data.ctrl[:] = np.zeros(model.nu)

    # Compute the solution
    qpos_idx = range(7)
    # poses = gen_random_pushing_poses(sim, qpos_idx, N=1000)
    poses = gen_random_pushing_poses(sim, qpos_idx, N=1000)
    # Visualize the solution
    print("Found {} poses".format(len(poses)))

    # Save.
    save_path = os.path.join(kuka_asset_dir(), 'random_pushing_poses')
    np.save(save_path, poses)

    viewer = mujoco_py.MjViewer(sim)
    while True:
        # import pdb; pdb.set_trace()
        sim.data.qpos[qpos_idx] = poses[np.random.choice(len(poses))]
        sim.forward()
        for i in range(100):
            # pass
            viewer.render()



