from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinJacobian, forwardKinSite
from gym_kuka_mujoco.utils.quaternion import mat2Quat
from gym_kuka_mujoco.envs.assets import kuka_asset_dir
import os
import mujoco_py
import numpy as np

# Get the model path
model_filename = 'full_kuka_no_collision.xml'
model_path = os.path.join(kuka_asset_dir(), model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

# The points to be transformed.
pos = np.array([0, 0, 0], dtype=np.float64)
quat = np.array([1, 0, 0, 0], dtype=np.float64)
body_id = model.body_name2id('kuka_link_7')

# Compute the forward kinematics
xpos, xrot = forwardKin(sim, pos, quat, body_id)
jacp, jacr = forwardKinJacobian(sim, pos, body_id)

print("Position:\n{}\nRotation:\n{}".format(xpos, xrot))
print("Position Jacobian:\n{}\nRotation Jacobian:\n{}".format(jacp.T, jacr.T))

## The peg tip position at the home position.
model_filename = 'full_peg_insertion_experiment_no_hole_no_gravity.xml'
model_path = os.path.join(kuka_asset_dir(), model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

sim.data.qpos[:] = np.array([np.pi/2,
                            -np.pi/6,
                            -np.pi/3,
                            -np.pi/2,
                             np.pi*3/4,
                            -np.pi/4,
                             0.0])

print(sim.data.qpos[:]s)

xpos, xrot = forwardKinSite(sim, 'peg_tip', recompute=True)
xquat = mat2Quat(xrot)
print("Position:\n{}\nRotation:\n{}".format(xpos, xquat))