import numpy as np
import cv2
#Import necessary functions

from python.matchPics import matchPics
from python.planarH import computeH_ransac
from python.helper import plotMatches
from python.planarH import compositeH
from python.loadVid import loadVid
opts = get_opts()
#Write script for Q3.1

# Load two videos

# book = loadVid('./data/book.mov')

# panda = loadVid('./data/ar_source.mov')
cv_cover = cv2.imread('./data/cv_cover.jpg',1)
cv_height = cv_cover.shape[0]
cv_width = cv_cover.shape[1]
#%% 测试部分
# see the matches
book_frame = book[1,:,:,:]
panda_frame = panda[1,:,:,:]
#

panda_frame = cv2.resize(panda_frame, (int(panda_frame.shape[1]*(cv_cover.shape[0]/panda_frame.shape[0])), cv_cover.shape[0]))
    

matches, locs1, locs2 = matchPics(book_frame,cv_cover, opts)
# plotMatches(book_frame, cv_cover, matches, locs1, locs2)

panda_crop = panda_frame[mid_height-cv_height//2:mid_height+cv_height//2, mid_width-cv_width//2:mid_width+cv_width//2,:]
#%%
# for each frame:
for i in range(min(book.shape[0], panda.shape[0])):
    book_frame = book[i,:,:,:]
    panda_frame = panda[i,:,:,:]
    
    # 将panda_frame 变得和cv_cover 一样大
   
    panda_frame = cv2.resize(panda_frame, (int(panda_frame.shape[1]*(cv_cover.shape[0]/panda_frame.shape[0])), cv_cover.shape[0]))
    
    # 将panda裁剪成cv_cover的尺寸
        #找到图片中心点
    mid_height = (panda_frame.shape[0]+1)//2
    mid_width = (panda_frame.shape[1]+1)//2
        #找到cv_cover的尺寸
    panda_crop = panda_frame[mid_height-cv_height//2:mid_height+cv_height//2, mid_width-cv_width//2:mid_width+cv_width//2,:]