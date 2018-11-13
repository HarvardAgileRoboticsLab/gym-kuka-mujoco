import mujoco_py
import numpy as np

def forwardKin(sim, pos, quat, body_id):
    '''
    Compute the forward kinematics for the position and orientation of a frame
    in a particular body.
    '''
    # Create buffers to store the result.
    xpos = np.zeros(3, dtype=np.float64)
    xrot = np.identity(3, dtype=np.float64).flatten()

    # Compute Kinematics and 
    mujoco_py.functions.mj_kinematics(sim.model, sim.data)
    mujoco_py.functions.mj_local2Global(sim.data, xpos, xrot, pos, quat, body_id);

    # Reshape the rotation matrix and return.
    xrot = xrot.reshape(3,3)
    return xpos, xrot
