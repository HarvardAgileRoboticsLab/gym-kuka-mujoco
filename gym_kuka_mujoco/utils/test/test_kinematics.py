from gym_kuka_mujoco.utils.kinematics import *
from gym_kuka_mujoco.utils.quaternion import subQuat, mat2Quat

import os
import mujoco_py


def test_forwardKinPosJacobian():
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
    xpos, _ = forwardKin(sim, pos, quat, body_id)
    jacp, _ = forwardKinJacobian(sim, pos, body_id)

    # Compute the kinematic jacobian with finite differences.
    jac_approx = np.zeros((3, 7))
    for i in range(7):
        dq = np.zeros(model.nq)
        dq[i] = 1e-6
        sim.data.qpos[:] = q_nom + dq
        xpos_, _ = forwardKin(sim, pos, quat, body_id)
        jac_approx[:, i] = (xpos_ - xpos) / 1e-6

    assert np.allclose(jac_approx - jacp, np.zeros_like(jacp), atol=1e-6)

def test_forwardKinRotJacobian():
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
    _, xrot = forwardKin(sim, pos, quat, body_id)
    xquat = mat2Quat(xrot)
    _, jacr = forwardKinJacobian(sim, pos, body_id)

    # Compute the kinematic jacobian with finite differences.
    jac_approx = np.zeros((3, 7))
    for i in range(7):
        dq = np.zeros(model.nq)
        dq[i] = 1e-6
        sim.data.qpos[:] = q_nom + dq
        _, xrot_ = forwardKin(sim, pos, quat, body_id)
        xquat_ = mat2Quat(xrot_)
        jac_approx[:, i] = subQuat(xquat_, xquat) / 1e-6

    assert np.allclose(jac_approx - jacr, np.zeros_like(jacr), atol=1e-6)

if __name__ == '__main__':
    test_forwardKinPosJacobian()
    test_forwardKinRotJacobian()
