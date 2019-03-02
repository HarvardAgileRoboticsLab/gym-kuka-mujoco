import numpy as np
import mujoco_py

import os
import mujoco_py
from gym_kuka_mujoco import kuka_asset_dir
from gym_kuka_mujoco.controllers import (DirectTorqueController, SACTorqueController, RelativeInverseDynamicsController, InverseDynamicsController, ImpedanceController, ImpedanceControllerV2, PDController, RelativePDController, controller_registry)

from tuned_control_params import controller_params

def compute_stiffness(controller, sim, qpos, action):
    nq = len(qpos)
    sim.data.qpos[:] = qpos
    sim.forward()
    controller.set_action(action)
    torque_0 = controller.get_torque()

    stiffness = np.zeros((nq, nq)) 
    for i in range(nq):
        # perturb the position
        eps = 1e-6
        d_qpos = np.zeros(nq)
        d_qpos[i] = eps
        sim.data.qpos[:] = qpos + d_qpos
        sim.forward()

        # rest the action
        torque_i = controller.get_torque()
        
        stiffness[:,i] = (torque_i - torque_0)/eps
    
    return stiffness

def compute_average_stiffness_forbenius_norm(controller_name, controller_options):
    model_filename = 'full_kuka_mesh_collision.xml'
    model_path = os.path.join(kuka_asset_dir(), model_filename)
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)

    controller_cls = controller_registry[controller_name]
    controller = controller_cls(sim, **controller_options)
      

    qpos_min = sim.model.jnt_range[:,0]
    qpos_max = sim.model.jnt_range[:,1]
    stiffnesses = []
    for i in range(10000):
        action = controller.action_space.sample()
        qpos = np.random.uniform(qpos_min, qpos_max)
        qpos = .5*np.ones_like(qpos_min)
        stiffnesses.append(compute_stiffness(controller, sim, qpos, action))

    norms = [np.linalg.norm(s) for s in stiffnesses]
    return np.median(norms)

def compute_average_stiffness_forbenius_norm_null_action(controller, sim):
    qpos_min = sim.model.jnt_range[:,0]
    qpos_max = sim.model.jnt_range[:,1]
    stiffnesses = []
    action_dim = len(controller.action_space.low)
    for i in range(10000):
        action = np.zeros(action_dim)
        qpos = np.random.uniform(qpos_min, qpos_max)
        qpos = .5*np.ones_like(qpos_min)
        stiffnesses.append(compute_stiffness(controller, sim, qpos, action))

    norms = [np.linalg.norm(s) for s in stiffnesses]
    return np.median(norms)

if __name__ == "__main__":
    for name, options in controller_params.items():
        if "Relative" in name or "Impedance" in name:
            stiffness = compute_average_stiffness_forbenius_norm(name, options)    
            print("Mean stiffness norm for the {}".format(name))
            print(stiffness)
