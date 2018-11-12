import os
import numpy as np
import mujoco_py
import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

model_path = os.path.join('..','..','envs','assets', 'full_kuka_mesh_collision.xml')
model = mujoco_py.load_model_from_path(model_path)
# model.integrator = 0
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

N = 1000
t = np.linspace(0,2*np.pi,N)
# reference trajectory
qpos_set = .1*np.sin(t)
qvel_set = .1*np.cos(t)

# allocate simulation trajectory
qpos_sim_pd = np.zeros((N,7))
qvel_sim_pd = np.zeros((N,7))
qpos_sim_spd = np.zeros((N-1,7))
qvel_sim_spd = np.zeros((N-1,7))
qpos_sim_spd_euler = np.zeros((N-1,7))
qvel_sim_spd_euler = np.zeros((N-1,7))

# pd gains
kp = 3
kd = np.array([0.03, 0.03, 0.01, 0.01, 0.005, 0.005, 0.0001])*2*np.sqrt(kp)

# kd = kp*model.opt.timestep

# Simulate with the causal PD controller
for i in range(N):
    qpos_sim_pd[i,:] = sim.data.qpos
    qvel_sim_pd[i,:] = sim.data.qvel
    ctrl = kp*(qpos_set[i] - sim.data.qpos) + kd*(qvel_set[i] - sim.data.qvel)
    sim.data.ctrl[:] = ctrl
    sim.step()
    viewer.render()

# Simulate with the non-causal PD controller
zero_ctrl = np.zeros(7)
sim.reset()
for i in range(N-1):
    qpos_sim_spd[i,:] = sim.data.qpos
    qvel_sim_spd[i,:] = sim.data.qvel
    sim.data.ctrl[:] = zero_ctrl
    
    sim.forward()
    ctrl = kp*(qpos_set[i+1] - sim.data.qpos - model.opt.timestep*sim.data.qvel) \
            + kd*(qvel_set[i+1] - sim.data.qvel - model.opt.timestep*sim.data.qacc)
    sim.data.ctrl[:] = ctrl
    sim.step()
    # viewer.render()

# Simulate with the non-causal PD controller and forward euler
zero_ctrl = np.zeros(7)
sim.reset()
for i in range(N-1):
    qpos_sim_spd_euler[i,:] = sim.data.qpos
    qvel_sim_spd_euler[i,:] = sim.data.qvel
    sim.data.ctrl[:] = zero_ctrl
    # import pdb; pdb.set_trace()
    sim.forward()
    ctrl = kp*(qpos_set[i+1] - sim.data.qpos - model.opt.timestep*sim.data.qvel) \
            + kd*(qvel_set[i+1] - sim.data.qvel - model.opt.timestep*sim.data.qacc)
    sim.data.ctrl[:] = ctrl
    sim.forward()
    sim.data.qpos[:] = sim.data.qpos + model.opt.timestep*sim.data.qvel
    sim.data.qvel[:] = sim.data.qvel + model.opt.timestep*sim.data.qacc
    # viewer.render()

# Plot the results.
plt.figure()
plt.plot(t, qpos_set)
plt.plot(t, qpos_sim_pd)

plt.figure()
plt.plot(t, qpos_set)
plt.plot(t[:-1], qpos_sim_spd)

plt.show()