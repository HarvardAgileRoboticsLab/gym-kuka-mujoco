import mujoco_py
import numpy as np
import os
from gym_kuka_mujoco.utils.quaternion import axisAngle2Quat, mulQuat, identity_quat
from gym_kuka_mujoco.utils.randomize import sample_pose

# Load model
model_filename = 'full_peg_insertion_experiment_moving_big_hole.xml'
model_filename = 'full_peg_insertion_experiment_moving_many_holes.xml'
model_path = os.path.join('..','..','gym_kuka_mujoco','envs','assets', model_filename)

# Load path
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Simulate while randomizing position
hole_pos_init = sim.data.get_mocap_pos('hole0').copy()
for i in range(100000):
    sim.data.ctrl[:] = 0.00*np.random.random(model.nu)

    sim_state = sim.get_state()
    sim_state.qpos[:7] = np.zeros(7)
    sim.set_state(sim_state)

    sim.step()
    viewer.render()
    for i in range(5):
        hole_pos, hole_quat = sample_pose(hole_pos_init, identity_quat)
        sim.data.set_mocap_pos('hole{}'.format(i), hole_pos)
        sim.data.set_mocap_quat('hole{}'.format(i), hole_quat)
    for i in range(5,15):
        sim.data.set_mocap_pos('hole{}'.format(i), np.zeros(3))

