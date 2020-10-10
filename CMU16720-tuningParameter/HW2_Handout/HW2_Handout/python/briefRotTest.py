import numpy as np
import cv2
from matchPics import matchPics
import matplotlib.pyplot as plt
import scipy.ndimage as sci
import os
import opts
#Q2.1.6
#Read the image and convert to grayscale, if necessary
#%%

img = cv2.imread('D:/Academic/CMU/Course/2020Fall/CV/Homework/HW2_Handout/HW2_Handout/data/cv_cover.jpg')
#%%
x = []
y = []
opts1 = opts.get_opts()
for i in range(36):
	#Rotate Image
    img_rotate = sci.rotate(img, i*10)

	#Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(img, img_rotate, opts1)
    
	#Update histogram
    # degree = i * 10
    x.append(i * 10)
    y.append(matches.shape[0])
    # match_count = matches.shape[0]

#Display histogram
plt.bar(x,y,5)
# plt.bar(x[0],y[0],5)
# plt.bar(x[1],y[1],5)
# plt.bar(x[2],y[2],5)
