import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
#Write script for Q2.2.4
opts = get_opts()

# Read cv_cover.jpg, cv_desk.png and hp_cover.jpg
cv_cover = cv2.imread('D:\Garbage\CMU16720-master\HW2_Handout\HW2_Handout\data\cv_cover.jpg',1)
cv_desk = cv2.imread('D:\Garbage\CMU16720-master\HW2_Handout\HW2_Handout\data\cv_desk.png',1)
hp_cover = cv2.imread('D:\Garbage\CMU16720-master\HW2_Handout\HW2_Handout\data\hp_cover.jpg',1)

#%%
matches, locs1, locs2 = matchPics(cv_desk,cv_cover, opts)
# plotMatches(cv_desk, cv_cover, matches, locs1, locs2)

locs1 = locs1[matches[:,0],:]
locs1[:,[0,1]] = locs1[:,[1,0]]

locs2 = locs2[matches[:,1],:]
locs2[:,[0,1]] = locs2[:,[1,0]]

result = computeH_ransac(locs1, locs2, opts)
H1 = result[0]
# # H1 = np.linalg.inv(H1)
img_new = cv2.warpPerspective(cv_cover, H1, (cv_desk.shape[1],cv_desk.shape[0]))
cv2.imshow('1',img_new)




# #%%
# matches, locs1, locs3 = matchPics(img1,img3, opts)
# # plotMatches(img1, img3, matches, locs1, locs3)

# locs1 = locs1[matches[:,0],:]
# locs1[:,[0,1]] = locs1[:,[1,0]]

# locs3 = locs3[matches[:,1],:]
# locs3[:,[0,1]] = locs3[:,[1,0]]

# result = computeH_ransac(locs1, locs3, opts)
# H2 = result[0]
# #%%
# H = H1 @ H2

# new_img = cv2.warpPerspective(img3, H,(locs3[0,0]*20,locs3[0,1]*20))
# cv2.imshow('1', new_img)