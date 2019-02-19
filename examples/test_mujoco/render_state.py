from gym_kuka_mujoco.utils.kinematics import forwardKin
from gym_kuka_mujoco.envs.assets import kuka_asset_dir
import os
import mujoco_py
import numpy as np

# Get the model path
model_filename = 'full_kuka_no_collision.xml'
model_filename = 'full_peg_insertion_experiment.xml'
model_filename = 'full_peg_insertion_experiment_moving_hole_id=050.xml'
model_path = os.path.join(kuka_asset_dir(), model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)


# Set the state and visualize.
# alpha = 3
# a = -.1
# b = -.35

alpha = 4
a = -.2
b = 0
sim.data.qpos[:] = np.array([0, np.pi/alpha + a, 0, -np.pi/alpha + a + b, 0, np.pi/alpha + b, 0])
sim.data.qpos[:] = np.array([-0.00336991,  0.76797655, -0.01091426, -1.11394698,  0.0388207,  1.3678605 , -0.0077345])
sim.data.qpos[:] = np.array([-1.03213825e-08,  6.68086145e-01,  1.85550710e-08, -1.12884164e+00,  -1.17948636e-08,  1.34466485e+00,  6.88901499e-09])
# sim.data.qpos[:] = np.zeros(7)
print(sim.data.qpos)
sim.forward()
while True:
    viewer.render()
