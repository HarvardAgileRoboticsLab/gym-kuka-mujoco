from gym_kuka_mujoco.utils.kinematics import forwardKin, inverseKin, identity_quat
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