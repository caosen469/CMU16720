# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:24:13 2020

@author: sihan
"""
import numpy as np
import cv2
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
def computeA(x1,x2):
    N = x1.shape[0]
    A = np.zeros((2*N,9))
    for i in range(x1.shape[0]):
        x = x2[i,0]
        y = x2[i,1]
        u = x1[i,0]
        v = x1[i,1]
        
        A[i*2,:] = np.array([[x,y,1,0,0,0,-x*u,-y*u,-u]])
        A[i*2+1,:] = np.array([[0,0,0,x,y,1,-x*v,-y*v,-v]])

#%%
def computeH(x1, x2):
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
  
    H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
    # H2to1 = (v[-1,:]).reshape(3,3)
    
    return H2to1

#%%
def computeH_norm(x1, x2):
    #Q2.2.2
    #%%
    # Translate to the mean
    # centroid1 = np.mean(x1, axis=0)
    
    # x1 = x1 - centroid1
    # distance_from_origin = np.sum(x1**2, axis=1)
    # max_distance = np.sqrt(np.max(distance_from_origin))
    # x1 = x1/max_distance*np.sqrt(2)
    
    # centroid2 = np.mean(x2, axis=0)
    
    # x2 = x2 - centroid2
    # distance_from_origin = np.sum(x2**2, axis=1)
    # max_distance = np.sqrt(np.max(distance_from_origin))
    # x2 = x2/max_distance*np.sqrt(2)
    #%%
    centroid1 = np.mean(x1, axis=0)
    x1_translation = x1 - centroid1
    distance_from_origin = np.sum(x1_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    k = np.sqrt(2)/max_distance
    # print('centorid is ', centroid1)
    # print('k is', k)
    tx_1 = -k * centroid1[0]
    ty_1 = -k * centroid1[1]
    print()
    print('x1 is', x1)
    T1 = np.array([[k,0,tx_1],[0,k,ty_1],[0,0,1]])
    # print(T1)
    padding = np.ones((x1.shape[0],1))
    x1 = np.append(x1, padding, axis=1)
    x1 = T1 @ x1.T
    x1 = x1.T[:,0:2]
    
    
    
    centroid2 = np.mean(x2, axis=0)
    # print('centroid2 is ', centroid2)
    x2_translation = x2 - centroid2
    distance_from_origin = np.sum(x2_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    # print('max distance is ', max_distance)
    k = np.sqrt(2)/max_distance
    
    tx_2 = -k * centroid2[0]
    ty_2 = -k * centroid2[1]
    
    T2 = np.array([[k,0,tx_2],[0,k,ty_2],[0,0,1]])
    print()
    # print('T1 is',T1)
    print()
    # print('T2 is',T2)
    padding = np.ones((x2.shape[0],1))
    x2 = np.append(x2, padding, axis=1)
    x2 = T2 @ x2.T
    x2 = x2.T[:,0:2]
    
  
    x1_input = x1
    x2_input = x2
    print('x1 input for H computation is ', x1_input)
    # print('x2 input for H computation is ', x2_input)
    # print('x2 input for H computation is ', x2_input)
    # print()
    #Similarity transform 2
    #%%
    H2to1 = computeH(x1_input, x2_input)
    print()
    print('H2to1 is ', H2to1)

    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
    #Denormalization
    print()
    print('After denormalization, H2to1 is ', H2to1)
    return H2to1,T1,T2
#%%
x1 = np.array([100,200,100,200],dtype=np.float32)
y1 = np.array([100,100,200,200],dtype=np.float32)

x2 = x1 - 100
y2 = y1 - 100

# x1 = np.array([459,473,403,403,405,434],dtype=np.float32)
# x2 = np.array([283,285,526,544,552,550])
# y2 = np.array([482,494,371,367,365,392])
u = np.column_stack((x1,y1))
x = np.column_stack((x2,y2))

result = computeH(x,x)
result1, T1, T2 = computeH_norm(x,x)