import mujoco_py
import numpy as np
import os
import time
from gym_kuka_mujoco.utils.quaternion import axisAngle2Quat, mulQuat, identity_quat
from gym_kuka_mujoco.utils.randomize import sample_pose
from gym_kuka_mujoco.utils.kinematics import forwardKin
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples_unrestricted
from gym_kuka_mujoco.envs.assets import kuka_asset_dir


def gen_random_reachable_holes(sim, hole_center, pos_range=0.8, angle_range=np.pi, num_samples=250):
    reachable_holes = []
    i = 0
    while i < num_samples:

        hole_pos, hole_quat = sample_pose(hole_center, identity_quat, pos_range=pos_range, angle_range=angle_range)
        sim.data.set_mocap_pos('hole', hole_pos)
        sim.data.set_mocap_quat('hole', hole_quat)
        
        try:
            qpos_sol = hole_insertion_samples_unrestricted(sim, nsamples=5, insertion_range=(0, 0.06), raise_on_fail=True)
        except RuntimeError:
            continue

        success = True
        for qpos in qpos_sol:
            sim_state = sim.get_state()
            sim_state.qpos[:7] = qpos.copy()
            sim_state.qvel[:] = np.zeros_like(sim_state.qvel[:])
            sim.set_state(sim_state)

            sim.forward()
            if sim.data.ncon > 0:
                print('There is a collision in the pose, discarding')
                success = False
                break

        if not success:
            continue
        
        print('Found a good pose!!!')
        reachable_holes.append({
            'hole_pos':hole_pos,
            'hole_quat':hole_quat,
            'good_poses':qpos_sol
            })
        i = i+1

    return reachable_holes


if __name__ == "__main__":

    model_filename = 'full_peg_insertion_experiment_moving_hole_id=050.xml'
    model_path = os.path.join(kuka_asset_dir(), model_filename)

    # Load path
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    sim.data.ctrl[:] = np.zeros(model.nu)

    # Get initial positions
    base_id = sim.model.body_name2id('kuka_base')
    hole_id = sim.model.body_name2id('hole')
    kuka_base_pos, _ = forwardKin(sim, np.array([0.,0.,0.]), identity_quat, base_id)
    hole_base_pos, _ = forwardKin(sim, np.array([0.,0.,0.]), identity_quat, hole_id)

    # # Generate with large randomness.
    # hole_center = kuka_base_pos + np.array([0., 0., .5])
    # reachable_holes = gen_random_reachable_holes(sim, pos_range=.8, angle_range=np.pi, num_samples=250)
    # print('Found {} reachable_holes'.format(len(reachable_holes)))

    # # Save.
    # save_path = os.path.join(kuka_asset_dir(), 'random_reachable_holes_large_randomness')
    # np.save(save_path, reachable_holes)


    # Generate with small randomness.
    hole_center = kuka_base_pos + np.array([0., 0., .5])
    reachable_holes = gen_random_reachable_holes(sim, hole_base_pos, pos_range=.1, angle_range=np.pi/4, num_samples=250)
    print('Found {} reachable_holes'.format(len(reachable_holes)))

    # Save.
    save_path = os.path.join(kuka_asset_dir(), 'random_reachable_holes_small_randomness')
    np.save(save_path, reachable_holes)





