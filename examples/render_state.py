from gym_kuka_mujoco.utils.kinematics import forwardKin
import os
import mujoco_py
import numpy as np

# Get the model path
model_filename = 'full_kuka_no_collision.xml'
model_filename = 'full_peg_insertion_experiment.xml'
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

alpha = 4
a = -.2
b = 0
sim.data.qpos[:] = np.array([0, np.pi/alpha + a, 0, -np.pi/alpha + a + b, 0, np.pi/alpha + b, 0])
# sim.data.qpos[:] = np.zeros(7)
print(sim.data.qpos)
sim.forward()
while True:
    viewer.render()
