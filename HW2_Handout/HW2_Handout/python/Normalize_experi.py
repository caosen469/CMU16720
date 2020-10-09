# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 02:27:16 2020

@author: sihan
"""
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def normalize(x1):
	#Q2.2.2
    
    #%%
    # length = x1.shape[0]
    # # sum_x1 = np.sum(x1, axis=0) 
    # # centroid1 = sum_x1 / length
    # centroid1 = np.mean(x1, axis=0)
    # # print('for x1, the centroid is ', centroid1)
    # s1 = np.sqrt(2)/((1/length)*np.sum((np.sum((x1-centroid1)**2,axis=1))**(0.5)))
    # t1_x = -s1 * centroid1[0]
    # t1_y = -s1 * centroid1[1]
    # # print()
    # # print('s1 is', s1)
    # T1 = np.array([[s1,0,t1_x],[0,s1,t1_y],[0,0,1]])
    # print(T1)
    
    # padding = np.zeros((x1.shape[0],1))
    # x1 = np.append(x1, padding, axis=1)
    # normalized_x1 = T1 @ x1.T
    # print(normalized_x1)
    # normalized_x1 = normalized_x1.T
    # normalized_x1 = normalized_x1[:,0:2]
    # return normalized_x1
    # # sum_x2 = np.sum(x2, axis=0)
    #%%
    centroid1 = np.mean(x1, axis=0)
    x1_translation = x1 - centroid1
    distance_from_origin = np.sum(x1_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    k = np.sqrt(1)/max_distance
    print()
    print(k)
    tx_1 = -k * centroid1[0]
    ty_1 = -k * centroid1[1]
    
    T1 = np.array([[k,0,tx_1],[0,k,ty_1],[0,0,1]])
    # print(T1)
    padding = np.ones((x1.shape[0],1))
    x1 = np.append(x1, padding, axis=1)
    print(x1)
    x1 = T1 @ x1.T
    print(x1)
    x1 = x1.T[:,0:2]
    print(x1)
    return x1
    #%%
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
    # return x1
    #%%
    
# x1 = np.array([[2,2],[3,4]])
# x1_norm = normalize(x1)
x1 = np.array([202,202,500,523,530,522])
y1 = np.array([459,473,403,403,405,434])
# x2 = np.array([283,285,526,544,552,550])
# y2 = np.array([482,494,371,367,365,392])
img1 = np.column_stack((x1,y1))
# img2= np.column_stack((x2,y2))
x1 = img1
x1 = normalize(x1)
# print(normalize(x1))