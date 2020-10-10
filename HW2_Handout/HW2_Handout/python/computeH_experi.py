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
# def computeH(x1, x2):
#     '''
#     INPUTS:
#         x1 and x2 - Each are size (2 x N) matrices of corresponding (x, y)'  
#                  coordinates between two images
#     OUTPUTS:
#      H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
#             equation
#     '''
    #%%
# def __calculate_homography_matrix(origin, dest):
#     # type: (np.ndarray, np.ndarray) -> np.ndarray
#     """
#     :param origin: start points for homography
#     :param dest: destination points for homography
#     :return: calculated homography matrix(3 x 3)
#     """
#     assert origin.shape == dest.shape

#     # 点を調整する（数値計算上重要）
#     origin, c1 = __normalize(origin)  # 変換元
#     dest, c2 = __normalize(dest)      # 変換先
#     # 線形法のための行列を作る。対応ごとに2つの行になる。
#     nbr_correspondences = origin.shape[1]
#     a = np.zeros((2 * nbr_correspondences, 9))
#     for i in range(nbr_correspondences):
#         a[2 * i] = [-origin[0][i], -origin[1][i], -1, 0, 0, 0, dest[0][i] * origin[0][i], dest[0][i] * origin[1][i],
#                     dest[0][i]]
#         a[2 * i + 1] = [0, 0, 0, -origin[0][i], -origin[1][i], -1, dest[1][i] * origin[0][i], dest[1][i] * origin[1][i],
#                         dest[1][i]]
#     u, s, v = np.linalg.svd(a)
#     homography_matrix = v[8].reshape((3, 3))
#     homography_matrix = np.dot(np.linalg.inv(c2), np.dot(homography_matrix, c1))
#     homography_matrix = homography_matrix / homography_matrix[2, 2]
#     return homography_matrix
    
    #%%
def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    x1 = x1.T
    x2 = x2.T
    N = x1.shape[1]
    x1 = np.vstack((x1,np.ones(N,dtype=int)))
    x2 = np.vstack((x2,np.ones(N,dtype=int)))

    A = np.zeros((2*N,9),dtype=int)
    #print(A)
    for i in range(N):
        x = x1[0,i]
        y = x1[1,i]
        u = x2[0,i]
        v = x2[1,i]

        A[(i*2)+1,:] = [ -u, -v, -1,  0, 0, 0,  x*u, x*v, x]
   
        A[(i*2),:] = [ 0, 0, 0, -u, -v, -1, y*u,  y*v, y]
   
    u,s,v = np.linalg.svd(A)
    
    H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
    # print('H2to1 is ', H2to1)
   
    return H2to1
#%%
def computeH_norm(x1, x2):
    #Q2.2.2
    #%%
    centroid1 = np.mean(x1, axis=0)
    x1_translation = x1 - centroid1
    distance_from_origin = np.sum(x1_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    k = np.sqrt(2)/max_distance
    
    tx_1 = -k * centroid1[0]
    ty_1 = -k * centroid1[1]
    
    T1 = np.array([[k,0,tx_1],[0,k,ty_1],[0,0,1]])
    # print(T1)
    padding = np.ones((x1.shape[0],1))
    x1 = np.append(x1, padding, axis=1)
    x1 = T1 @ x1.T
    x1 = x1.T[:,0:2]
    
    centroid2 = np.mean(x2, axis=0)
    x2_translation = x2 - centroid2
    distance_from_origin = np.sum(x2_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    k = np.sqrt(2)/max_distance
    
    tx_2 = -k * centroid2[0]
    ty_2 = -k * centroid2[1]
    
    T2 = np.array([[k,0,tx_2],[0,k,ty_2],[0,0,1]])
    # print(T1)
    padding = np.ones((x2.shape[0],1))
    x2 = np.append(x2, padding, axis=1)
    x2 = T2 @ x2.T
    x2 = x2.T[:,0:2]
    x1_input = x1
    x2_input = x2
  
    #Similarity transform 2
    #%%
    H2to1 = computeH(x1_input, x2_input)
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
    #Denormalization
    print()
    print('H2to1 is ', H2to1)
    return H2to1 

#%%
#%%

# x1 = np.array([100,200,100,200],dtype=np.float32)
y1 = np.array([100,100,200,200],dtype=np.float32)

x1 = np.array([459,473,403,403],dtype=np.float32)
x2 = np.array([283,285,526,544])
y2 = np.array([482,494,371,367])
x = np.column_stack((x1,x1))
u = np.column_stack((x2,y2))

result = computeH(x,x)