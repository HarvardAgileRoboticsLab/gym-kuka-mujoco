from gym_kuka_mujoco.utils.kinematics import forwardKin, inverseKin, identity_quat
from gym_kuka_mujoco.utils.quaternion import mat2Quat
import numpy as np

def hole_insertion_samples(sim, nsamples=10, range=(0, 0.05)):
    # The points to be transformed.
    pos = np.array([0., 0., 0.])
    peg_body_id = sim.model.body_name2id('peg')
    tip_site_id = sim.model.site_name2id('peg_tip')
    tip_body_pos = sim.model.site_pos[tip_site_id]

    # The desired world coordinates
    hole_id = sim.model.body_name2id('hole')
    world_pos_desired, _ = forwardKin(sim, np.zeros(3), identity_quat, hole_id)
    world_pos_delta = np.zeros((nsamples, 3))
    world_pos_delta[:,2] = np.linspace(range[0], range[1], nsamples)
    world_pos_desired = world_pos_delta + world_pos_desired
    world_quat = np.array([0., 1., 0., 0.])

    # Compute the forward kinematics
    q_nom = np.zeros(7)
    q_init = np.zeros(7)
    upper = np.array([1e-6, np.inf, 1e-6, np.inf, 1e-6, np.inf, np.inf])
    lower = -upper

    q_sol = []
    for w_pos in world_pos_desired:
        q_opt = inverseKin(sim, q_init, q_nom, tip_body_pos, w_pos, world_quat, peg_body_id, upper=upper, lower=lower)
        q_sol.append(q_opt)

    return q_sol

def hole_insertion_samples_unrestricted(sim, nsamples=10, insertion_range=(0, 0.05), raise_on_fail=False):
    # The points to be transformed.
    pos = np.array([0., 0., 0.])
    peg_body_id = sim.model.body_name2id('peg')
    tip_site_id = sim.model.site_name2id('peg_tip')
    tip_body_pos = sim.model.site_pos[tip_site_id]

    # The desired world coordinates
    hole_id = sim.model.body_name2id('hole')

    hole_pos_delta = np.zeros((nsamples, 3))
    hole_pos_delta[:,2] = np.linspace(insertion_range[0], insertion_range[1], nsamples)

    world_pos_desired = []
    world_quat_desired = []
    world_quat = np.array([0., 1., 0., 0.])
    for i in range(nsamples):
        pos_desired, mat_desired = forwardKin(sim, hole_pos_delta[i,:], world_quat, hole_id)
        world_pos_desired.append(pos_desired)
        world_quat_desired.append(mat2Quat(mat_desired))

    # Compute the forward kinematics
    q_nom = np.zeros(7)
    q_init = np.zeros(7)
    upper = sim.model.jnt_range[:, 1]
    lower = sim.model.jnt_range[:, 0]

    q_sol = []
    for w_pos, w_quat in zip(world_pos_desired, world_quat_desired):
        q_opt = inverseKin(sim, q_init, q_nom, tip_body_pos, w_pos, w_quat, peg_body_id, upper=upper, lower=lower, raise_on_fail=raise_on_fail)
        q_sol.append(q_opt)

    return q_sol