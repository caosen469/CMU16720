import numpy as np
import cv2
import skimage.io 
import skimage.color
from python.opts import get_opts

#Import necessary functions
from python.matchPics import matchPics
from python.planarH import computeH_ransac
from python.helper import plotMatches
from python.planarH import compositeH
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
#Write script for Q2.2.4
opts = get_opts()

# Read cv_cover.jpg, cv_desk.png and hp_cover.jpg
cv_cover = cv2.imread('./data/cv_cover.jpg',1)
cv_desk = cv2.imread('./data/cv_desk.png',1)
hp_cover = cv2.imread('./data/hp_cover.jpg',1)
#%%
matches, locs1, locs2 = matchPics(cv_desk,cv_cover, opts)
# plotMatches(cv_desk, cv_cover, matches, locs1, locs2)
# resize the hp_cover to the same size of cv_cover
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]))

locs1 = locs1[matches[:,0],:]
locs1[:,[0,1]] = locs1[:,[1,0]]

locs2 = locs2[matches[:,1],:]
locs2[:,[0,1]] = locs2[:,[1,0]]

result = computeH_ransac(locs1, locs2, opts)
H1 = result[0]

composite_img = compositeH(H1, hp_cover, cv_desk)
# # H1 = np.linalg.inv(H1)
# img_new = cv2.warpPerspective(hp_cover, H1, (cv_desk.shape[1],cv_desk.shape[0]))
# # cv2.imshow('1',img_new)

# mask = cv2.inRange(img_new,0,255)
# mask = cv2.bitwise_not(mask)
# # img1_bg =  cv2.bitwise_and(cv_desk,cv_desk,mask = mask)
# img1_bg =  cv2.bitwise_and(cv_desk,cv_desk,mask = cv2.bitwise_not(mask))
# # cv2.imshow('result',img1_bg)
# composite_img = cv2.add(img1_bg,img_new)
# cv2.imshow('composite',composite_img)

