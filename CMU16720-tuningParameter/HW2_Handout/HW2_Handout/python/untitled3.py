# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:03:29 2020

@author: sihan
"""
import numpy as np
def computeH(x1, x2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    p1 = x1.T
    p2 = x2.T
    assert(p1.shape[1] == p2.shape[1])
    assert(p1.shape[0] == 2)
    num_points = p1.shape[1]
    A = np.zeros((2 * num_points, 9))
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 0]
    ), 0:3] = np.hstack((p2.T, np.ones((num_points, 1))))
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 1]
    ), 3:6] = np.hstack((p2.T, np.ones((num_points, 1))))
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 0]
    ), 6:9] = -np.hstack((p2.T, np.ones((num_points, 1)))) * p1.T[:, 0].reshape(-1, 1)
    A[np.array(
        [row for row in range(2 * num_points) if row % 2 == 1]
    ), 6:9] = -np.hstack((p2.T, np.ones((num_points, 1)))) * p1.T[:, 1].reshape(-1, 1)
    V = np.dot(A.T, A)
    eigen_values, eigen_vecs = np.linalg.eigh(V)
    squeezed_H = eigen_vecs[:, 0]
    H2to1 = squeezed_H.reshape(3, 3)
    return H2to1

x1 = np.array([[2, 2],[3,3],[4,4],[5, 5],[6,6]])
x2 = np.array([[2, 2],[3,3],[4,4],[5, 5],[6,6]])

H = computeH(x1,x2)