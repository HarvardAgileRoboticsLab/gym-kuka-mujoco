import numpy as np
from gym_kuka_mujoco.utils.quaternion import mulQuat, axisAngle2Quat

def sample_pose(pos_init, quat_init, pos_range=.2, angle_range=.2):
    # Sample a change in position.
    high_pos = pos_range*np.ones(3)
    low_pos = -high_pos
    d_pos = np.random.uniform(low_pos, high_pos)

    # Sample a change in orientation.
    axis = np.random.uniform(-1, 1, 3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-angle_range, angle_range)
    d_quat = axisAngle2Quat(axis, angle)

    # Apply the translation and rotation.
    pos = pos_init + d_pos
    quat = mulQuat(d_quat, quat_init)

    return pos, quat