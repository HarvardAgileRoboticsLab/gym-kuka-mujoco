import numpy as np
import mujoco_py
from gym_kuka_mujoco.utils.quaternion import *

def test_random_quat():
    q = random_quat()
    assert q.shape == (4,), 'quaternion must be length 4'
    assert np.allclose(np.linalg.norm(q), 1.), 'quaternion must have unit norm'

def test_subQuat():
    q = random_quat()

    v = subQuat(q, identity_quat)
    v_ = quat2Vel(q)
    assert np.allclose(v, v_), 'subQuat test failed'

def test_quatInegrate():
    q2 = random_quat()
    v = subQuat(q2, identity_quat)
    q2_ = quatIntegrate(identity_quat, v)
    assert np.allclose(q2, q2_), 'quatIntegrate test failed'

    q1 = random_quat()
    q2 = random_quat()
    v = subQuat(q2, q1)
    qv = quatIntegrate(identity_quat, v)
    q2_ = mulQuat(qv, q1)
    assert np.allclose(q2, q2_), 'quatIntegrate test failed'

def test_quatAdd():
    q1 = random_quat()
    q2 = random_quat()
    v = subQuat(q2, q1)
    q2_ = quatAdd(q1, v)
    assert np.allclose(q2, q2_)

def test_rotVecQuat():
    q = random_quat()
    v = np.random.random(3)
    mat = quat2Mat(q)
    v_rot = mat.dot(v)
    v_rot_ = rotVecQuat(v, q)
    assert np.allclose(v_rot, v_rot_)


if __name__ == '__main__':
    test_random_quat()
    test_subQuat()
    test_quatInegrate()
    test_quatAdd()
    test_rotVecQuat()
    print('tests passed')