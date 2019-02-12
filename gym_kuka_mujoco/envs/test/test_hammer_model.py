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

def test_nail_model():
    model_path = os.path.join(kuka_asset_dir(), 'hammer', 'full_nail_only.xml')
    render_mujoco(model_path)

def test_nail_model_friction():
    model_path = os.path.join(kuka_asset_dir(), 'hammer', 'full_nail_only.xml')
    qpos = np.zeros(1)
    qvel = -1*np.ones(1)
    sim_mujoco(model_path, qpos, qvel)

def test_full_hammer_model():
    model_path = os.path.join(kuka_asset_dir(), 'full_hammer_experiment_no_gravity.xml')
    model = mujoco_py.load_model_from_path(model_path)

    joint_idx = [model.joint_name2id('kuka_joint_{}'.format(i)) for i in range(1,8)]
    nail_idx = model.joint_name2id('nail_position')
    
    qpos = np.zeros(8)
    qvel = np.zeros(8)
    qpos[joint_idx] = np.array([0, -.5, 0, -2, 0, 0, 0])
    qvel[joint_idx] = np.array([0, 0, 0, 0, 0, 20, 0])
    sim_mujoco(model_path, qpos, qvel)

def test_full_hammer_model_no_collision():
    model_path = os.path.join(kuka_asset_dir(), 'full_hammer_experiment_no_collision_no_gravity.xml')
    model = mujoco_py.load_model_from_path(model_path)

    joint_idx = [model.joint_name2id('kuka_joint_{}'.format(i)) for i in range(1,8)]
    nail_idx = model.joint_name2id('nail_position')
    
    qpos = np.zeros(8)
    qvel = np.zeros(8)
    qpos[joint_idx] = np.array([0, -.5, 0, -2, 0, 0, 0])
    qvel[joint_idx] = np.array([0, 0, 0, 0, 0, 20, 0])
    sim_mujoco(model_path, qpos, qvel)

if __name__ == "__main__":
    # test_nail_model()
    # test_nail_model_friction()
    test_full_hammer_model()
    # test_full_hammer_model_no_collision()