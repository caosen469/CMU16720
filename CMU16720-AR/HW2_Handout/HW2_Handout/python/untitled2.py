# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:24:55 2020

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
    N = p1.shape[1]
    p1 = np.vstack((p1,np.ones(N,dtype=int)))
    p2 = np.vstack((p2,np.ones(N,dtype=int)))

    # A matrix
    A = np.zeros((2*N,9),dtype=int)
    #print(A)
    for i in range(N):
        x = p1[0,i]
        y = p1[1,i]
        u = p2[0,i]
        v = p2[1,i]
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

x1 = np.array([[2, 2],[3,3],[4,4],[5, 5],[6,6]])
x2 = np.array([[2, 2],[3,3],[4,4],[5, 5],[6,6]])
H2to1 = computeH(x1,x2)