import mujoco_py
import numpy as np
import scipy.optimize
from gym_kuka_mujoco.utils.quaternion import identity_quat, mat2Quat, subQuat

identity_quat = np.array([1., 0., 0., 0.])

def forwardKin(sim, pos, quat, body_id, recompute=True):
    '''
    Compute the forward kinematics for the position and orientation of a frame
    in a particular body.
    '''
    # Create buffers to store the result.
    xpos = np.zeros(3, dtype=np.float64)
    xrot = np.identity(3, dtype=np.float64).flatten()

    # Compute Kinematics and
    if recompute: 
        mujoco_py.functions.mj_kinematics(sim.model, sim.data)
    mujoco_py.functions.mj_local2Global(sim.data, xpos, xrot, pos, quat, body_id, False)

    # Reshape the rotation matrix and return.
    xrot = xrot.reshape(3,3)
    return xpos, xrot

def forwardKinSite(sim, site_name, recompute=True):
    '''
    Compute the forward kinematics for the position and orientation a labelled site.
    '''
    # Compute Kinematics and return data.
    if recompute:
        mujoco_py.functions.mj_kinematics(sim.model, sim.data)

    if type(site_name) is list:
        xpos = [sim.data.get_site_xpos(n) for n in site_name]
        xrot = [sim.data.get_site_xmat(n) for n in site_name]
    else:
        xpos = sim.data.get_site_xpos(site_name)
        xrot = sim.data.get_site_xmat(site_name)

    return xpos, xrot

def forwardKinJacobian(sim, pos, body_id, recompute=True):
    '''
    Compute the forward kinematics for the position and orientation of a frame
    in a particular body.
    '''
    # Create buffers to store the result.
    jac_shape = (3, sim.model.nv)
    jacp = np.zeros(jac_shape, dtype=np.float64).flatten()
    jacr = np.zeros(jac_shape, dtype=np.float64).flatten()

    # Create buffers to store the global frame.
    xpos = np.zeros(3, dtype=np.float64)
    xrot = np.identity(3, dtype=np.float64).flatten()

    # Compute Kinematics and
    if recompute: 
        sim.forward()
    mujoco_py.functions.mj_local2Global(sim.data, xpos, xrot, pos, identity_quat, body_id, False)
    mujoco_py.functions.mj_jac(sim.model, sim.data, jacp, jacr, xpos, body_id)

    # Reshape the jacobian matrices and return.
    jacp = jacp.reshape(jac_shape)
    jacr = jacr.reshape(jac_shape)
    
    return jacp, jacr

def forwardKinJacobianSite(sim, site_id, recompute=True):
    '''
    Compute the forward kinematics for the position and orientation of the
    frame attached to a particular site.
    '''

    if type(site_id) is str:
        site_id = sim.model.site_name2id(site_id)

    # Create buffers to store the result.
    jac_shape = (3, sim.model.nv)
    jacp = np.zeros(jac_shape, dtype=np.float64).flatten()
    jacr = np.zeros(jac_shape, dtype=np.float64).flatten()

    # Compute Kinematics and
    if recompute: 
        sim.forward()
    mujoco_py.functions.mj_jacSite(sim.model, sim.data, jacp, jacr, site_id)

    # Reshape the jacobian matrices and return.
    jacp = jacp.reshape(jac_shape)
    jacr = jacr.reshape(jac_shape)
    
    return jacp, jacr


def inverseKin(sim, q_init, q_nom, body_pos, world_pos, world_quat, body_id, reg=1e-4, upper=None, lower=None, cost_tol=1e-6, raise_on_fail=False, qpos_idx=None):
    '''
    Use SciPy's nonlinear least-squares method to compute the inverse kinematics
    '''

    if qpos_idx is None:
        qpos_idx = range(len(q_init))

    def residuals(q):
        sim.data.qpos[qpos_idx] = q
        xpos, xrot = forwardKin(sim, body_pos, identity_quat, body_id)
        quat = mat2Quat(xrot)
        q_diff = subQuat(quat, world_quat)
        res = np.concatenate((xpos-world_pos, q_diff, reg*(q-q_nom)))
        return res

    def jacobian(q):
        sim.data.qpos[qpos_idx] = q
        jacp, jacr = forwardKinJacobian(sim, body_pos, body_id)
        residual_jacobian = np.vstack((jacp, jacr, reg*np.identity(q.size)))
        return residual_jacobian

    if lower is None:
        lower = sim.model.jnt_range[qpos_idx,0]
    else:
        lower = np.maximum(lower, sim.model.jnt_range[qpos_idx,0])

    if upper is None:
        upper = sim.model.jnt_range[qpos_idx,1]
    else:
        upper = np.minimum(upper, sim.model.jnt_range[qpos_idx,1])

    result = scipy.optimize.least_squares(residuals, q_init, bounds=(lower, upper))
    # result = scipy.optimize.least_squares(residuals, q_init, jac=jacobian, bounds=(lower, upper))

    if not result.success:
        print("Inverse kinematics failed with status: {}".format(result.status))
        if raise_on_fail:
            raise RuntimeError

    if result.cost > cost_tol:
        print("Inverse kinematics failed to find a sufficiently low cost")
        if raise_on_fail:
            raise RuntimeError("Infeasible")
    return result.x
