# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:24:13 2020

@author: sihan
"""
import numpy as np
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
        
