import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    A = np.empty((0,9))
    for each in range(x1.shape[0]):
        row1 = np.array([[x2[each,0],x2[each, 1],1, 0, 0, 0,-x2[each,0]*x1[each,0], -x2[each,1]*x1[each,0],-x1[each, 0]]])
        A = np.append(A, row1, axis=0)
        row2 = np.array([[0,0,0,x2[each,0],x2[each,1],1, -x2[each,0]*x1[each,1],-x2[each,1]*x1[each,1],-x1[each,1]]])
    

    D, V = np.linalg.eig(A.T @ A)
    index = np.argwhere(D == 0)[0][0]
    
    H2to1_vector = V[: index]
    H2to1 = H2to1_vector.reshape((3,3))

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
    
    #%%
	#Compute the centroid of the points
    # length = x1.shape[0]
    # sum_x1 = np.sum(x1, axis=0) 
    # centroid1 = sum_x1 / length
    centroid1 = np.mean(x1, axis=0)
    
    s1 = np.sqrt(2)/((1/length)*np.sum(np.sum((x1-centroid1)**2, axis=1)))
    t1_x = -s1 * centroid1[0]
    t1_y = -s1 * centroid1[1]
    
    T1 = np.array([[s1,0,t1_x],[0,s1,t1_y],[0,0,1]])
    
    # sum_x2 = np.sum(x2, axis=0)
    centroid2 = np.mean(x2, axis=0)
    s2 = np.sqrt(2)/((1/length)*np.sum(np.sum((x2-centroid2)**2, axis=1)))
    t2_x = -s2 * centroid2[0]
    t2_y = -s2 * centroid2[1]
    
        length = x1.shape[0]
    sum_x1 = np.sum(x1, axis=0) 
    centroid1 = sum_x1 / length
    
    s1 = np.sqrt(2)/((1/length)*np.sum(np.sum((x1-centroid1)**2, axis=1)))
    t1_x = -s1 * centroid1[0]
    t1_y = -s1 * centroid1[1]
    
    T1 = np.array([[s1,0,t1_x],[0,s1,t1_y],[0,0,1]])
    # s1 = np.sqrt(2)/((1/length)*np.sum((np.sqrt(abs(x1 - np.mean(x1,axis=0))**2))))
#%%    
	#Shift the origin of the points to the centroid
    x1 = x1 - centroid1
    x2 = x2 - centroid2
    

    # T1 = np.array([[1,0,-centroid1[0]],[0,1,-centroid1[1]],[0,0,1]])
    # T2 = np.array([[1,0,-centroid2[0]],[0,1,-centroid2[1]],[0,0,1]])
#%%    
    # turn x1, x2 to homogeneous coordinate
    padding1 = np.zeros((x1.shape[0],1))
    x1_homo = np.append(x1, padding, axis=1)
    # x1_homo = x1_homo.T
    # x1_norm = x1_homo @ T1
    
    padding2 = np.zeros((x2.shape[0],1))
    x2_homo = np.append(x2, padding, axis=1)
    # x2_homo = x2_homo.T
    # x2_norm = x2_homo @ T2
    

#%%
    
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    
#%%

	#Similarity transform 1


	#Similarity transform 2


	#Compute homography


	#Denormalization
	

	return H2to1
    T2 = np.array([[s2,0,t2_x],[0,s2,t2_y],[0,0,1]])


	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    padding1 = np.ones((x1.shape[0],1))
    padding2 = np.ones((x2.shape[0],1))
    x1_homo = np.append(x1, padding1, axis=1)
    x2_homo = np.append(x2, padding2, axis=1)

	#Similarity transform 1


	#Similarity transform 2


	#Compute homography


	#Denormalization
	

	return H2to1

#%%

def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	


	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	
	return composite_img

# [240, 163],
# plotMatches(img,img_rotate,matches, locs1, locs2)
# cv2.circle(img,(locs1[240,:][0],locs1[240,:][1]),5,(0,0,0),4)