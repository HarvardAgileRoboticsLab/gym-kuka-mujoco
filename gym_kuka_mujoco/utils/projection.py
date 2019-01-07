import numpy as np 
from .quaternion import quat2Mat

def project(x,y):
    '''
    Decomposes x into the parallel and perpendicular components to y.
    '''
    vector = vector/np.linalg.norm(vector)
    P = np.einsum('i,j->ij',vector,vector)
    proj = P.dot(x)
    perp = (np.eye(3) - P).dot(x)

    return proj, perp

def rotate_cost_by_matrix(Q,mat):
    '''
    Rotates a 3 dimensional cost matrix by a rotation matrix.
    '''
    return mat.T.dot(Q).dot(mat)

def rotate_cost_by_quaternion(Q,quat):
    '''
    Rotates a 3 dimensional cost matrix by a quaternion
    '''
    mat = quat2Mat(quat)
    return rotate_cost_by_matrix(Q, mat)