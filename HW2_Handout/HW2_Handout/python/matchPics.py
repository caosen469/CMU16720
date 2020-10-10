import numpy as np
import cv2
import skimage.color
from python.helper import briefMatch
from python.helper import computeBrief
from python.helper import corner_detection

def matchPics(I1, I2, opts):
    #I1, I2 : Images to match
    #opts: input opts
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

#%%
    #Convert Images to GrayScale
    # I1 = cv2.imread(I1, 0)
    I1 = skimage.color.rgb2gray(I1)
    
    I2 = skimage.color.rgb2gray(I2)
    # I2 = cv2.imread(I2, 0)

    #%%
    #Detect Features in Both Images
    # print('Here is good')
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)
    
    
    #%%
    #Obtain descriptors for the computed feature locations
    
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)
    
    # Change the locs to the corresponding1
    # locs1 = locs1[matches[:,0],:]
    # locs2 = locs2[matches[:,1],:]
    #%%
    return matches, locs1, locs2
