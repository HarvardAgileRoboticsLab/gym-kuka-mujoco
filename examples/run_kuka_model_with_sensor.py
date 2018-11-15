import time
import mujoco_py
import numpy as np
import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import os

model_filename = 'full_peg_insertion_experiment.xml'
model_path = os.path.join('..','gym_kuka_mujoco','envs','assets', model_filename)

model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

print("Simulating at :{}".format(model.opt.timestep))
t_start = time.time()
Nsteps = 1000
sensor_data = np.zeros((Nsteps, 6))
for i in range(Nsteps):
    sim.data.ctrl[:] = 0.00*np.random.random(model.nu)
    sim.step()
    sensor_data[i,:] = sim.data.sensordata
    viewer.render()

t_end = time.time()
simulation_time = 1000*model.opt.timestep

plt.plot(sensor_data)
plt.show()