import mujoco_py
import numpy as np
import scipy.optimize

identity_quat = np.array([1., 0., 0., 0.])

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
    mujoco_py.functions.mj_local2Global(sim.data, xpos, xrot, pos, quat, body_id)

    # Reshape the rotation matrix and return.
    xrot = xrot.reshape(3,3)
    return xpos, xrot

def forwardKinJacobian(sim, pos, body_id):
    '''
    Compute the forward kinematics for the position and orientation of a frame
    in a particular body.
    '''
    # Create buffers to store the result.
    jac_shape = (3, sim.model.nv)
    jacp = np.zeros(jac_shape, dtype=np.float64).flatten()
    jacr = np.zeros(jac_shape, dtype=np.float64).flatten()

    # Create buffers to store the result.
    xpos = np.zeros(3, dtype=np.float64)
    xrot = np.identity(3, dtype=np.float64).flatten()

    # Compute Kinematics and 
    sim.forward()
    mujoco_py.functions.mj_local2Global(sim.data, xpos, xrot, pos, identity_quat, body_id)
    mujoco_py.functions.mj_jac(sim.model, sim.data, jacp, jacr, xpos, body_id)

    # Reshape the rotation matrix and return.
    jacp = jacp.reshape(jac_shape)
    jacr = jacr.reshape(jac_shape)
    
    return jacp, jacr

def inverseKin(sim, q_init, q_nom, body_pos, world_pos, body_id, reg=1e-2):
    '''
    Use SciPy's nonlinear least-squares method to compute the inverse kinematics
    '''

    def residuals(q):
        sim.data.qpos[:] = q
        xpos, _ = forwardKin(sim, body_pos, identity_quat, body_id)
        res = np.concatenate((xpos-world_pos, reg*(q-q_nom)))
        # import pdb; pdb.set_trace()
        return res

    def jacobian(q):
        sim.data.qpos[:] = q
        jacp, _ = forwardKinJacobian(sim, body_pos, body_id)
        residual_jacobian = np.vstack((jacp, reg*np.identity(q.size)))
        return residual_jacobian

    lower = sim.model.jnt_range[:,0]
    upper = sim.model.jnt_range[:,1]

    result = scipy.optimize.least_squares(residuals, q_init, jac=jacobian, bounds=(lower, upper))

    if not result.success:
        print("Inverse kinematics failed with status: {}".format(result.status))

    return result.x
