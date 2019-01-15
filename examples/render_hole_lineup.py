from gym_kuka_mujoco.utils.kinematics import forwardKin
import os
import mujoco_py
import numpy as np

# Get the model path
model_filename = 'hole_lineup.xml'
model_path = os.path.join('..', 'gym_kuka_mujoco', 'envs', 'assets',
                          model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)


# Set the state and visualize.
# alpha = 3
# a = -.1
# b = -.35

while True:
    viewer.render()
