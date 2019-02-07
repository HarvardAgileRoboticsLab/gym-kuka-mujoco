import mujoco_py
import numpy as np
import os
import time
from gym_kuka_mujoco.utils.quaternion import axisAngle2Quat, mulQuat, identity_quat
from gym_kuka_mujoco.utils.randomize import sample_pose
from gym_kuka_mujoco.utils.kinematics import forwardKin
from gym_kuka_mujoco.utils.insertion import hole_insertion_samples_unrestricted
from gym_kuka_mujoco.envs.assets import kuka_asset_dir

model_filename = 'full_peg_insertion_experiment_moving_hole_id=050.xml'
model_path = os.path.join(kuka_asset_dir(), model_filename)

# Load path
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
sim.data.ctrl[:] = np.zeros(model.nu)

# Simulate while randomizing position
sim.model.body_name2id('kuka_base')
kuka_base_pos, _ = forwardKin(sim, np.array([0.,0.,0.]), identity_quat, 2)
hole_center = kuka_base_pos + np.array([0., 0., .5])

reachable_holes = []

for i in range(4000):

    hole_pos, hole_quat = sample_pose(hole_center, identity_quat, pos_range=.8, angle_range=np.pi)
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

print('Found {} reachable_holes'.format(len(reachable_holes)))

save_path = os.path.join(kuka_asset_dir(), 'random_reachable_holes')
np.save(save_path, reachable_holes)