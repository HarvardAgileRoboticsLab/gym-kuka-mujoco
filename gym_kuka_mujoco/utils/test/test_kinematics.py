from gym_kuka_mujoco.utils.kinematics import *

import os
import mujoco_py


def test_forwardKinJacobian():
    # Build the model path.
    model_filename = 'full_kuka_no_collision.xml'
    model_path = os.path.join('..', '..', 'envs', 'assets', model_filename)

    # Construct the model and simulation objects.
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)

    # Set an arbitrary state.
    q_nom = np.array([.1, .2, .3, .4, .5, .6, .7])
    sim.data.qpos[:] = q_nom

    # Choose an arbitrary point and a link with a dense jacobian.
    body_id = model.body_name2id('kuka_link_7')
    pos = np.array([.1, .2, .3])
    quat = np.array([1., 0., 0., 0.])

    # Compute the forward kinematics and kinematic jacobian.
    xpos, xrot = forwardKin(sim, pos, quat, body_id)
    jacp, _ = forwardKinJacobian(sim, pos, body_id)

    # Compute the kinematic jacobian with finite differences.
    jac_approx = np.zeros((3, 7))
    for i in range(7):
        dq = np.zeros(model.nq)
        dq[i] = 1e-6
        sim.data.qpos[:] = q_nom + dq
        xpos_, xrot_ = forwardKin(sim, pos, quat, body_id)
        jac_approx[:, i] = (xpos_ - xpos) / 1e-6

    assert np.allclose(jac_approx - jacp, np.zeros_like(jacp), atol=1e-6)

if __name__ == '__main__':
    test_forwardKinJacobian()
