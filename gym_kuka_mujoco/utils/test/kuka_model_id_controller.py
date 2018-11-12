import os
import numpy as np
import mujoco_py
import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

model_path = os.path.join('..','..','envs','assets', 'full_kuka_mesh_collision.xml')
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# 1000 knot points between 0 and 2*pi
t = np.arange(0, 2*np.pi, model.opt.timestep)
N = len(t)

# reference trajectory
qpos_set = 1*np.sin(t)**2
qvel_set = 1*2*np.cos(t)*np.sin(t)

# allocate simulation trajectory
qpos_sim_id = np.zeros((N,7))
qvel_sim_id = np.zeros((N,7))
ctrl_sim_id = np.zeros((N,7))

# pd gains for inverse dynamics
kp_id = 100
kd_id = 2*np.sqrt(kp_id)

# pd gains for wrapping controller
# proportional gains are scaled by the mass of subtree that it is moving around
# derivative gains get the minimum of the value required to keep the simulation stable and the critical damping gain
kp_pd = 1e-2*model.body_subtreemass[2:]
kd_pd = np.minimum(kp_pd*model.opt.timestep, 2*np.sqrt(model.body_subtreemass[2:]*kp_pd))

# Simulate with the ID controller
for i in range(N):
    qpos_sim_id[i,:] = sim.data.qpos
    qvel_sim_id[i,:] = sim.data.qvel
    sim.data.qacc[:] = kp_id*(qpos_set[i] - sim.data.qpos) + kd_id*(qvel_set[i] - sim.data.qvel)
    pd_torque = kp_pd*(qpos_set[i] - sim.data.qpos) + kd_pd*(qvel_set[i] - sim.data.qvel)
    mujoco_py.functions.mj_inverse(model,sim.data)
    sim.data.ctrl[:] = sim.data.qfrc_inverse[:]/300 + pd_torque
    ctrl_sim_id[i,:] = sim.data.ctrl
    print(sim.data.ctrl[:])
    sim.step()
    viewer.render()


# Plot the results.
plt.figure()
plt.plot(t, qpos_set)
plt.plot(t, qpos_sim_id)

plt.figure()
plt.plot(t, ctrl_sim_id)


plt.show()