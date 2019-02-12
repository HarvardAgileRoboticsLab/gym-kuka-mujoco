import os
import time
import mujoco_py
import numpy as np

from gym_kuka_mujoco.envs.assets import kuka_asset_dir

def render_mujoco(model_path):
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    while True:
        viewer.render()
        time.sleep(.01)

def sim_mujoco(model_path, qpos, qvel):
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.forward()
    while True:
        sim.step()
        viewer.render()
        time.sleep(.01)

def test_valve_model():
    model_path = os.path.join(kuka_asset_dir(), 'valve', 'full_valve_only.xml')
    render_mujoco(model_path)

def test_valve_model_friction():
    model_path = os.path.join(kuka_asset_dir(), 'valve', 'full_valve_only.xml')
    qpos = np.zeros(1)
    qvel = 100*np.ones(1)
    sim_mujoco(model_path, qpos, qvel)

def test_full_valve_model():
    model_path = os.path.join(kuka_asset_dir(), 'full_valve_experiment_no_gravity.xml')
    render_mujoco(model_path)

if __name__ == "__main__":
    # test_valve_model()
    # test_valve_model_friction()
    test_full_valve_model()