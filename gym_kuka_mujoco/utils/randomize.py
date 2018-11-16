import numpy as np
from gym_kuka_mujoco.utils.quaternion import mulQuat, axisAngle2Quat

def sample_pose(pos_init, quat_init):
    # Sample a change in position.
    d_pos = np.random.uniform(np.array([-.2,-.2,-.2]), np.array([.2,.2,.2]))

    # Sample a change in orientation.
    axis = np.random.uniform(-1, 1, 3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-1, 1)
    d_quat = axisAngle2Quat(axis, angle)

    # Apply the translation and rotation.
    pos = pos_init + d_pos
    quat = mulQuat(d_quat, quat_init)

    return pos, quat