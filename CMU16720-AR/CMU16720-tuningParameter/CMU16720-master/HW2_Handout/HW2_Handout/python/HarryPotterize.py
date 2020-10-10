import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches

#Write script for Q2.2.4
opts = get_opts()

# Read cv_cover.jpg, cv_desk.png and hp_cover.jpg
img1 = cv2.imread('D:/Academic/CMU/Course/2020Fall/CV/Homework/HW2_Handout/HW2_Handout/data\cv_cover.jpg',1)
img2 = cv2.imread('D:/Academic/CMU/Course/2020Fall/CV/Homework/HW2_Handout/HW2_Handout/data\cv_desk.png',1)
img3 = cv2.imread('D:/Academic/CMU/Course/2020Fall/CV/Homework/HW2_Handout/HW2_Handout/data\hp_cover.jpg',1)

matches, locs1, locs2 = matchPics(img1,img1, opts)
locs1 = locs1[matches[:,0],:]
locs2 = locs2[matches[:,1],:]
plotMatches(img1, img2, matches, locs1, locs2)

result = computeH_ransac(locs2, locs1, opts)
