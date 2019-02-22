from gym_kuka_mujoco.utils.kinematics import inverseKin
from gym_kuka_mujoco.utils.quaternion import identity_quat
from gym_kuka_mujoco.envs.assets import kuka_asset_dir
import os
import mujoco_py
import numpy as np

# Get the model path
model_filename = 'full_pushing_experiment_no_gravity.xml'
model_path = os.path.join(kuka_asset_dir(), model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

# The points to be transformed.
pos = np.array([0., 0., 0.])
body_id = model.body_name2id('peg')

# Compute the forward kinematics
q_nom = np.zeros(7)
q_init = np.random.random(7)
peg_tip_idx = model.site_name2id('peg_tip')
body_pos = model.site_pos[peg_tip_idx]
# world_pos = np.array([0.7, 0., 1.22]) # above the block
world_pos = np.array([0.775, -0.02, 1.20]) # to the +x and -y of the block
world_quat = np.array([0, 1., 0, 0])
qpos_idx = range(7)
q_opt = inverseKin(sim, q_init, q_nom, body_pos, world_pos, world_quat, body_id, qpos_idx=qpos_idx)

# Visualize the solution
print("Optimal pose: {}\n".format(q_opt))

sim.data.qpos[qpos_idx] = q_opt
sim.forward()

viewer = mujoco_py.MjViewer(sim)
while True:
    viewer.render()
