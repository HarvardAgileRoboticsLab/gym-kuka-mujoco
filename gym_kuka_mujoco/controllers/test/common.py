import os
import mujoco_py
from gym_kuka_mujoco import kuka_asset_dir

def create_sim(collision = False):
    if collision:
        model_filename = 'full_kuka_mesh_collision.xml'
    else:
        model_filename = 'full_kuka_no_collision.xml'
    model_path = os.path.join(kuka_asset_dir(), model_filename)
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    return sim
