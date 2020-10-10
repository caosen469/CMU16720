# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 02:27:16 2020

@author: sihan
"""
import numpy as np
def normalize(x1):
	#Q2.2.2
    
    #%%
	#Compute the centroid of the points
    length = x1.shape[0]
    sum_x1 = np.sum(x1, axis=0) 
    centroid1 = sum_x1 / length
    
    s1 = np.sqrt(2)/((1/length)*np.sum(np.sum((x1-centroid1)**2, axis=1)))
    t1_x = -s1 * centroid1[0]
    t1_y = -s1 * centroid1[1]
    
    T1 = np.array([[s1,0,t1_x],[0,s1,t1_y],[0,0,1]])
    return T1

x1 = np.array([202,202,500,523,530,522])
y1 = np.array([459,473,403,403,405,434])
x2 = np.array([283,285,526,544,552,550])
y2 = np.array([482,494,371,367,365,392])
img1 = np.column_stack((x1,y1))
img2= np.column_stack((x2,y2))
x1 = img1
T1 = normalize(x1)
print(normalize(x1))