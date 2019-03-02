# System imports

# Package imports
import mujoco_py
from mujoco_py.generated import const
import numpy as np

# Local imports
from gym_kuka_mujoco.controllers import ImpedanceControllerV2
from gym_kuka_mujoco.utils.kinematics import forwardKinSite
from gym_kuka_mujoco.utils.quaternion import mat2Quat, quat2Mat
from common import create_sim

def render_frame(viewer, pos, quat):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])
    mat = quat2Mat(quat)
    cylinder_half_height = 0.02
    pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, .005, cylinder_half_height],
                      mat=mat)

def render_point(viewer, pos):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])

def vis_impedance_fixed_setpoint(collision=False):
    options = dict()
    options['model_path'] = 'full_kuka_no_collision.xml'
    options['rot_scale'] = .3
    options['stiffness'] = np.array([10.,10.,10.,10.,10.,10.])

    sim = create_sim(collision=collision)
    controller = ImpedanceControllerV2(sim, **options)

    viewer = mujoco_py.MjViewer(sim)
    for i in range(10):

        # Set a random state to get a random feasible setpoint.
        qpos = np.random.uniform(-1., 1, size=7)
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()

        controller.set_action(np.zeros(6))

        # Set a different random state and run the controller.
        qpos = np.random.uniform(-1., 1., size=7)
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()

        for i in range(3000):
            sim.data.ctrl[:] = controller.get_torque()
            sim.step()
            render_frame(viewer, controller.pos_set, controller.quat_set)
            viewer.render()

def vis_impedance_random_setpoint(collision=False):
    options = dict()
    options['model_path'] = 'full_kuka_no_collision.xml'
    options['rot_scale'] = .3
    options['stiffness'] = np.array([1.,1.,1.,3.,3.,3.])

    sim = create_sim(collision=collision)
    controller = ImpedanceControllerV2(sim, **options)

    frame_skip = 50
    high = np.array([.1, .1, .1, 2, 2, 2])
    low = -np.array([.1, .1, .1, 2, 2, 2])

    viewer = mujoco_py.MjViewer(sim)
    for i in range(10):

        # Set a different random state and run the controller.
        qpos = np.random.uniform(-1., 1., size=7)
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()

        for i in range(3000):
            controller.set_action(np.random.uniform(high, low))
            for i in range(frame_skip):
                sim.data.ctrl[:] = controller.get_torque()
                sim.step()
                render_frame(viewer, controller.pos_set, controller.quat_set)
                viewer.render()

if __name__ == '__main__':
    # vis_impedance_fixed_setpoint()
    vis_impedance_fixed_setpoint(collision=True)
    # vis_impedance_random_setpoint()
    # vis_impedance_random_setpoint(collision=True)