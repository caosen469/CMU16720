# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 03:36:43 2020

@author: sihan
"""
import numpy as np
# def computeH(x1, x2):
#     '''
#     INPUTS:
#         x1 and x2 - Each are size (2 x N) matrices of corresponding (x, y)'  
#                  coordinates between two images
#     OUTPUTS:
#      H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
#             equation
#     '''
#     assert(x1.shape[1] == x2.shape[1])
#     assert(x1.shape[0] == 2)
#     num_points = x1.shape[1]
#     A = np.zeros((2 * num_points, 9))
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 0]
#     ), 0:3] = np.hstack((x2.T, np.ones((num_points, 1))))
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 1]
#     ), 3:6] = np.hstack((x2.T, np.ones((num_points, 1))))
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 0]
#     ), 6:9] = -np.hstack((x2.T, np.ones((num_points, 1)))) * x1.T[:, 0].reshape(-1, 1)
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 1]
#     ), 6:9] = -np.hstack((x2.T, np.ones((num_points, 1)))) * x1.T[:, 1].reshape(-1, 1)
#     V = np.dot(A.T, A)
#     eigen_values, eigen_vecs = np.linalg.eigh(V)
#     squeezed_H = eigen_vecs[:, 0]
#     H2to1 = squeezed_H.reshape(3, 3)
#     return H2to1
#%%
def computeH(x1, x2):
    '''
    INPUTS:
        x1 and x2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''

    #############################
    # TO DO ...
    # append ones to bottom row of x1, x2 so [x,y,1]'
    x1 = x1.T
    x2 = x2.T
    N = x1.shape[1]
    x1 = np.vstack((x1,np.ones(N,dtype=int)))
    x2 = np.vstack((x2,np.ones(N,dtype=int)))

    # A matrix
    A = np.zeros((2*N,9),dtype=int)
    #print(A)
    for i in range(N):
        x = x1[0,i]
        y = x1[1,i]
        u = x2[0,i]
        v = x2[1,i]
        #print(i)
        # odd - x rows
        A[(i*2)+1,:] = [ -u, -v, -1,  0, 0, 0,  x*u, x*v, x]
        #print(A)
        #even - y rows
        A[(i*2),:] = [ 0, 0, 0, -u, -v, -1, y*u,  y*v, y]
        #print(A)

    u,s,v = np.linalg.svd(A)
    #print(A)
    #w,v = np.linalg.eigh(np.matmul(np.transpose(A), A))
    #H2to1 = v[:,0].reshape(3,3)
    H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
    #H2to1 = np.transpose(v[:,0].reshape(3,3))
    return H2to1
#%%

x1 = np.array([100,200,100,200],dtype=np.float32)
y1 = np.array([100,100,200,200],dtype=np.float32)

# x1 = np.array([459,473,403,403,405,434],dtype=np.float32)
# x2 = np.array([283,285,526,544,552,550])
# y2 = np.array([482,494,371,367,365,392])
x = np.column_stack((x1,y1))

result = computeH(x,x)