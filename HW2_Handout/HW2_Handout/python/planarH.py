import numpy as np
import cv2
import numpy.matlib

def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    x1 = x1.T
    x2 = x2.T
    N = x1.shape[1]
    x1 = np.vstack((x1,np.ones(N,dtype=int)))
    x2 = np.vstack((x2,np.ones(N,dtype=int)))

    A = np.zeros((2*N,9),dtype=int)
    #print(A)
    for i in range(N):
        x = x1[0,i]
        y = x1[1,i]
        u = x2[0,i]
        v = x2[1,i]

        A[(i*2),:] = [ -u, -v, -1,  0, 0, 0,  x*u, x*v, x]
   
        A[(i*2)+1,:] = [ 0, 0, 0, -u, -v, -1, y*u,  y*v, y]
   
    u,s,v = np.linalg.svd(A)
    if v[-1,-1]==0:
        v[-1,-1] += 0.00000001
    H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
    # H2to1 = (v[-1,:]).reshape(3,3)
    # H2to1 = cv2.findHomography(x2,x1)
    # print('H2to1 is ', H2to1)
   
    return H2to1

#%%
def computeH_norm(x1, x2):
    #Q2.2.2
    #%%
    centroid1 = np.mean(x1, axis=0)
    x1_translation = x1 - centroid1
    distance_from_origin = np.sum(x1_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    k = np.sqrt(2)/max_distance
    
    tx_1 = -k * centroid1[0]
    ty_1 = -k * centroid1[1]
    
    T1 = np.array([[k,0,tx_1],[0,k,ty_1],[0,0,1]])
    # print(T1)
    padding = np.ones((x1.shape[0],1))
    x1 = np.append(x1, padding, axis=1)
    x1 = T1 @ x1.T
    x1 = x1.T[:,0:2]
    
    centroid2 = np.mean(x2, axis=0)
    x2_translation = x2 - centroid2
    distance_from_origin = np.sum(x2_translation**2, axis=1)
    max_distance = np.sqrt(np.max(distance_from_origin))
    k = np.sqrt(2)/max_distance
    
    tx_2 = -k * centroid2[0]
    ty_2 = -k * centroid2[1]
    
    T2 = np.array([[k,0,tx_2],[0,k,ty_2],[0,0,1]])
    # print(T1)
    padding = np.ones((x2.shape[0],1))
    x2 = np.append(x2, padding, axis=1)
    x2 = T2 @ x2.T
    x2 = x2.T[:,0:2]
    
    x1_input = x1
    x2_input = x2
  
    #Similarity transform 2
    #%%
    H2to1 = computeH(x1_input, x2_input)
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
    #Denormalization
    # print()
    # print('H2to1 is ', H2to1)
    return H2to1 

#%%

def computeH_ransac(locs1, locs2, opts):
    """
    locs1 and locs2 are coordinates that retrived from matches
    """
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # inlier_tol = opts.inlier_tol # the tolerance value for considering 
    inlier_tol=opts.inlier_tol
    current_inlier = 0
    inliers = np.zeros((locs1.shape[0],1))
    bestH2to1 = np.zeros((3,3))
    
    num_inlier = np.zeros(max_iters)
    H_record = [None] * max_iters
    inliers_record = np.empty((locs1.shape[0], 0))
    for i in range(max_iters):
    # for i in range(1):
        
    # 选出四个match。来计算homo
        # locs = np.append(locs1,locs2, axis=1)
        # np.random.shuffle(locs)
        # locs_1 = locs[0:4,0:2]
        # locs_2 = locs[0:4,2:4]
        if (locs1.shape[0]<4):
            continue
        randomIndex = np.random.choice(locs1.shape[0],4,replace=False)
        locs_1 = locs1[randomIndex, :]
        locs_2 = locs2[randomIndex, :]
        #%% 目前问题在这里
        H2to1 = computeH_norm(locs_1, locs_2)
        # print()
        # print('The H computation so far is ', H2to1)
        #%%
        # print(H2to1)
        # 计算locs2 映射到locs1 上
        padding = np.ones((locs2.shape[0],1))
        locs2_1 = np.append(locs2, padding, axis=1)
        locs2to1 = H2to1 @ locs2_1.T 
        # print()
        # print('The third coordinate', locs2to1[2,:])
        L = np.matlib.repmat(locs2to1[2,:],2,1)
        if np.count_nonzero(L) < L.shape[0]*L.shape[1]:
            L += 0.0000001
        locs2to1 = np.divide(locs2to1[0:2,:],L)
        locs2to1 = locs2to1.T
        
        
        # 计算误差
        tempInliersNum = 0
        
        curr_inliers = np.zeros((locs1.shape[0],1))
        for j in range(locs1.shape[0]):
            # print(locs2to1.shape)
            # print(locs1.shape)
            currDist = np.linalg.norm(locs2to1[j,:]-locs1[j,:])
            # print()
            # print(currDist)
            
            if currDist < inlier_tol:
                tempInliersNum += 1
                curr_inliers[j,0] = 1
        inliers_record = np.append(inliers_record, curr_inliers,axis=1)
        num_inlier[i] = tempInliersNum
        H_record[i] = H2to1
        
        # error_distance = np.linalg.norm(locs2to1 - locs1)
        # Bias = locs2to1 - locs1
        # 计算片差距离
        # error_distance = np.linalg.norm(Bias, ord=2, axis=1)
        # print()
        # print('error distance is', error_distance)
        # 统计inlier
        # index_inlier = np.where(error_distance<inlier_tol)
     
    
    bestH2to1 = H_record[np.argmax(num_inlier)]
    print('Inliers shape', inliers.shape)
    print()
    print('argmax is', np.argmax(num_inlier))
    inliers = inliers_record[:,np.argmax(num_inlier)]
        # if index_inlier[0].shape[0]>current_inlier:
        #      inliers = index_inlier
        #      bestH2to1 = H2to1
    return bestH2to1, inliers

#%%
def compositeH(H2to1, template, img):

    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    
    #x_template = H2to1*x_photo
    
    #For warping the template to the image, we need to invert it.
    
    
    

    # composite_img = cv2.warpPerspective(img, template)
    img_new = cv2.warpPerspective(template, H2to1, (img.shape[1],img.shape[0]))
    mask = cv2.inRange(img_new,0,255)
    mask = cv2.bitwise_not(mask)
    
    img_bg = cv2.bitwise_and(img, img, mask = cv2.bitwise_not(mask))
    
    composite_img = cv2.add(img_bg, img_new)
    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image
    
    return composite_img

