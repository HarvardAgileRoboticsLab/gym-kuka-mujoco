from gym_kuka_mujoco.utils.kinematics import forwardKin, inverseKin, identity_quat

import os
import mujoco_py
import numpy as np

# Get the model path
model_filename = 'full_peg_insertion_experiment.xml'
model_path = os.path.join('..', 'gym_kuka_mujoco', 'envs', 'assets',
                          model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# The points to be transformed.
pos = np.array([0., 0., 0.])
peg_body_id = model.body_name2id('peg')
tip_site_id = model.site_name2id('peg_tip')
tip_body_pos = model.site_pos[tip_site_id]

# The desired world coordinates
hole_id = model.body_name2id('hole')
world_pos_desired, _ = forwardKin(sim, np.zeros(3), identity_quat, hole_id)
world_pos_delta = np.zeros((10, 3))
world_pos_delta[:,2] = np.linspace(0,.075,10)
world_pos_desired = world_pos_delta + world_pos_desired
world_quat = np.array([0., 1., 0., 0.])

# Compute the forward kinematics
q_nom = np.zeros(7)
q_init = np.zeros(7)
upper = np.array([1e-6, np.inf, 1e-6, np.inf, 1e-6, np.inf, np.inf])
lower = -upper

q_sol = []
for w_pos in world_pos_desired:
    q_opt = inverseKin(sim, q_init, q_nom, tip_body_pos, w_pos, world_quat, peg_body_id, upper=upper, lower=lower)
    q_sol.append(q_opt)

while True:
    # Iterate through all of the solutions
    for q in q_sol:
        sim.data.qpos[:] = q
        sim.forward()
        viewer.render()
