import numpy as np
import cv2
#Import necessary functions

from python.matchPics import matchPics
from python.planarH import computeH_ransac
from python.helper import plotMatches
from python.planarH import compositeH
from python.loadVid import loadVid
from python.opts import get_opts
opts = get_opts()
#Write script for Q3.1

# Load two videos

book = loadVid('./data/book.mov')

panda = loadVid('./data/ar_source.mov')
cv_cover = cv2.imread('./data/cv_cover.jpg',1)
cv_height = cv_cover.shape[0]
cv_width = cv_cover.shape[1]
#%% 测试部分
# see the matches
# book_frame = book[1,:,:,:]
# panda_frame = panda[1,:,:,:]
# #

# panda_frame = cv2.resize(panda_frame, (int(panda_frame.shape[1]*(cv_cover.shape[0]/panda_frame.shape[0])), cv_cover.shape[0]))
    

# matches, locs1, locs2 = matchPics(book_frame,cv_cover, opts)
# # plotMatches(book_frame, cv_cover, matches, locs1, locs2)

# panda_crop = panda_frame[mid_height-cv_height//2:mid_height+cv_height//2, mid_width-cv_width//2:mid_width+cv_width//2,:]

# locs1 = locs1[matches[:,0],:]
# locs1[:,[0,1]] = locs1[:,[1,0]]

# locs2 = locs2[matches[:,1],:]
# locs2[:,[0,1]] = locs2[:,[1,0]]

# result = computeH_ransac(locs1, locs2, opts)
# H1 = result[0]

# composite_img = compositeH(H1, panda_crop, book_frame)
#%% Create Video Writer
 
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (book.shape[2],book.shape[3]),)
#%%
# for each frame:
for i in range(min(book.shape[0], panda.shape[0])):
# for i in range(1):
    book_frame = book[i,:,:,:]
    panda_frame = panda[i,:,:,:]
    
    # 将panda_frame 变得和cv_cover 一样大
   
    panda_frame = cv2.resize(panda_frame, (int(panda_frame.shape[1]*(cv_cover.shape[0]/panda_frame.shape[0])), cv_cover.shape[0]))
    
    matches, locs1, locs2 = matchPics(book_frame,cv_cover, opts)
    locs1 = locs1[matches[:,0],:]
    locs1[:,[0,1]] = locs1[:,[1,0]]

    locs2 = locs2[matches[:,1],:]
    locs2[:,[0,1]] = locs2[:,[1,0]]

    result = computeH_ransac(locs1, locs2, opts)
    H1 = result[0]
    # 将panda裁剪成cv_cover的尺寸
        #找到图片中心点
    mid_height = (panda_frame.shape[0]+1)//2
    mid_width = (panda_frame.shape[1]+1)//2
        #找到cv_cover的尺寸
    panda_crop = panda_frame[mid_height-cv_height//2:mid_height+cv_height//2, mid_width-cv_width//2:mid_width+cv_width//2,:]
    


    composite_img = compositeH(H1, panda_crop, book_frame)
    
    # cv2.imshow('1',composite_img)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    out.write(composite_img)
    
out.release()

