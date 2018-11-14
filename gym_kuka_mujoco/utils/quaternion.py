import numpy as np
import mujoco_py

identity_quat = np.array([1., 0., 0., 0.])

def mat2Quat(mat):
    '''
    Convenience function for mju_mat2Quat.
    '''
    res = np.zeros(4)
    mujoco_py.functions.mju_mat2Quat(res, mat.flatten())
    return res

def subQuat(qb, qa):
    '''
    Convenience function for mju_subQuat.
    '''
    # Allocate memory
    qa_t = np.zeros(4)
    q_diff = np.zeros(4)
    res = np.zeros(3)

    # Compute the subtraction
    mujoco_py.functions.mju_negQuat(qa_t, qa)
    mujoco_py.functions.mju_mulQuat(q_diff, qb, qa_t)
    mujoco_py.functions.mju_quat2Vel(res, q_diff, 1.)

    #   Mujoco 1.50 doesn't support the subQuat function. Uncomment this when
    #   mujoco_py upgrades to Mujoco 2.0
    # mujoco_py.functions.mju_subQuat(res, qa, qb)
    return res