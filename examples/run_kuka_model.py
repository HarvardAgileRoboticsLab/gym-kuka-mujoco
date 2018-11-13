import time
import mujoco_py
import numpy as np
import os

# model_filename = 'full_kuka_no_collision.xml'
# model_filename = 'full_kuka_mesh_collision.xml'
model_filename = 'full_peg_insertion_experiment.xml'

model_path = os.path.join('..','gym_kuka_mujoco','envs','assets', model_filename)

model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

print("Simulating at :{}".format(model.opt.timestep))
t_start = time.time()
for i in range(100000):
    sim.data.ctrl[:] = 0.00*np.random.random(model.nu)
    sim.step()
    viewer.render()
    # time.sleep(.1)
    
t_end = time.time()
simulation_time = 1000*model.opt.timestep
print(simulation_time/(t_end - t_start))