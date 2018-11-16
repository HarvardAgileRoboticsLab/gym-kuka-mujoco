import mujoco_py
import numpy as np
import os
from gym_kuka_mujoco.utils.quaternion import axisAngle2Quat, mulQuat, identity_quat
from gym_kuka_mujoco.utils.random import sample_pose

# Load model
model_filename = 'full_peg_insertion_experiment_moving_big_hole.xml'
model_path = os.path.join('..','gym_kuka_mujoco','envs','assets', model_filename)

# Load path
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Simulate while randomizing position
hole_pos_init = sim.data.get_mocap_pos('hole').copy()
for i in range(100000):
    sim.data.ctrl[:] = 0.00*np.random.random(model.nu)
    sim.step()
    viewer.render()
    hole_pos, hole_quat = sample_pose(hole_pos_init, identity_quat)
    sim.data.set_mocap_pos('hole', hole_pos)
    sim.data.set_mocap_quat('hole', hole_quat)
