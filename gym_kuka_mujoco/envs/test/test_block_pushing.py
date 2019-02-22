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

def test_block_model():
    model_path = os.path.join(kuka_asset_dir(), 'pushing', 'full_block_only.xml')
    render_mujoco(model_path)

def test_full_pushing_model():
    model_path = os.path.join(kuka_asset_dir(), 'full_pushing_experiment.xml')
    model = mujoco_py.load_model_from_path(model_path)

    joint_idx = [model.joint_name2id('kuka_joint_{}'.format(i)) for i in range(1,8)]
    nail_idx = model.joint_name2id('block_position')

def test_full_pushing_model_no_gravity():
    model_path = os.path.join(kuka_asset_dir(), 'full_pushing_experiment_no_gravity.xml')
    model = mujoco_py.load_model_from_path(model_path)

    joint_idx = [model.joint_name2id('kuka_joint_{}'.format(i)) for i in range(1,8)]
    nail_idx = model.joint_name2id('block_position')
    

    qpos = np.zeros(14)
    qvel = np.zeros(13)
    qpos[7:14] = np.array([.7, 0, 1.2, 1, 0, 0, 0])
    # qpos[7:14] = np.array([0.7, 0.1, 1.2, 0.92387953251, 0, 0, 0.38268343236])
    qpos[joint_idx] = np.array([0, 0, 0, -2, 0, .9, 0])
    sim_mujoco(model_path, qpos, qvel)

if __name__ == "__main__":
    # test_block_model()
    # test_full_pushing_model()
    test_full_pushing_model_no_gravity()
