from gym_kuka_mujoco.utils.insertion import hole_insertion_samples
import os
import mujoco_py

# Get the model path
model_filename = 'full_peg_insertion_experiment.xml'
model_path = os.path.join('..','..', 'gym_kuka_mujoco', 'envs', 'assets',
                          model_filename)

# Construct the model and simulation objects.
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)


q_sol = hole_insertion_samples(sim, nsamples=20)

while True:
    # Iterate through all of the solutions
    for q in q_sol:
        sim.data.qpos[:] = q
        sim.forward()
        viewer.render()
    for q in q_sol[::-1]:
        sim.data.qpos[:] = q
        sim.forward()
        viewer.render()
