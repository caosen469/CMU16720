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
        A = np.append(A, row2, axis=0)
    # print()
    # print('x1 is ', x1)
    # print()
    # print('x2 is ', x2)
    # print()
    # print('The value of A is', A)
#%%
    # D, V = np.linalg.eig(A.T @ A)
    # print()
    # print('The eigen values is ', D)
    # print()
    # print('The eigen vector is ', V)
    # # index = np.argwhere(D == 0)[0][0]
    # # index = np.argmin(np.abs(D))
    # index = np.argmin(D)
    
    # H2to1_vector = V[:, index]
    #%%
    [U,S,V] = np.linalg.svd(A.T@A)
    H2to1_vector = V.T[:,8]
    #%%
    H2to1 = H2to1_vector.reshape((3,3))
    return H2to1
#%%

# def computeH(x1, x2):
#     #Q2.2.1
#     #Compute the homography between two sets of points
#     N = x1.shape[0]
#     A = np.zeros((2*N, 9))
#     for i in range(N):
#         x = x2[i, 0]
#         y = x2[i, 1]
#         u = x1[i, 0]
#         v = x1[i, 1]
#         A[(i*2)+1,:]=[x,y,1,0,0,0,-x*u,-y*u,-u]
#         A[(i*2),:]=[0,0,0,x,y,1,-x*v,-y*v,-v]
        

#     #%%
#     print()
#     print('A is ', A)
#     [U,S,V] = np.linalg.svd(A.T@A)
#     # H2to1_vector = V.T[:,8]
#     #%%
#     # H2to1 = H2to1_vector.reshape((3,3))
#     H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
#     return H2to1
#%%
# def computeH(x1, x2):
#     '''
#     INPUTS:
#         p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
#                  coordinates between two images
#     OUTPUTS:
#      H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
#             equation
#     '''
#     p1 = x1.T
#     p2 = x2.T
#     assert(p1.shape[1] == p2.shape[1])
#     assert(p1.shape[0] == 2)
#     num_points = p1.shape[1]
#     A = np.zeros((2 * num_points, 9))
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 0]
#     ), 0:3] = np.hstack((p2.T, np.ones((num_points, 1))))
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 1]
#     ), 3:6] = np.hstack((p2.T, np.ones((num_points, 1))))
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 0]
#     ), 6:9] = -np.hstack((p2.T, np.ones((num_points, 1)))) * p1.T[:, 0].reshape(-1, 1)
#     A[np.array(
#         [row for row in range(2 * num_points) if row % 2 == 1]
#     ), 6:9] = -np.hstack((p2.T, np.ones((num_points, 1)))) * p1.T[:, 1].reshape(-1, 1)
#     V = np.dot(A.T, A)
#     eigen_values, eigen_vecs = np.linalg.eigh(V)
#     squeezed_H = eigen_vecs[:, 0]
#     H2to1 = squeezed_H.reshape(3, 3)
#     return H2to1
#%%

# def computeH(x1, x2):
#     '''
#     INPUTS:
#         p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
#                   coordinates between two images
#     OUTPUTS:
#       H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
#             equation
#     '''
#     p1 = x1.T
#     p2 = x2.T
#     N = p1.shape[1]
#     p1 = np.vstack((p1,np.ones(N,dtype=int)))
#     p2 = np.vstack((p2,np.ones(N,dtype=int)))

#     # A matrix
#     A = np.zeros((2*N,9),dtype=int)
#     print('p1 is ', p1)
#     #print(A)
#     for i in range(N):
#         x = p1[0,i]
#         y = p1[1,i]
#         u = p2[0,i]
#         v = p2[1,i]
#         #print(i)
#         # odd - x rows
#         A[(i*2)+1,:] = [ -u, -v, -1,  0, 0, 0,  x*u, x*v, x]
#         #print(A)
#         #even - y rows
#         A[(i*2),:] = [ 0, 0, 0, -u, -v, -1, y*u,  y*v, y]
#         #print(A)

#     u,s,v = np.linalg.svd(A)
#     print()
#     print(A)
#     #w,v = np.linalg.eigh(np.matmul(np.transpose(A), A))
#     #H2to1 = v[:,0].reshape(3,3)
#     H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
#     #H2to1 = np.transpose(v[:,0].reshape(3,3))
#     return H2to1

#%%


def computeH_norm(x1, x2):
    #Q2.2.2
    
    #%%
    #Compute the centroid of the points
    # print('x1 is', x1)
    # print('x2 is', x2)
    length = x1.shape[0]
    # sum_x1 = np.sum(x1, axis=0) 
    # centroid1 = sum_x1 / length
    centroid1 = np.mean(x1, axis=0)
    # print('for x1, the centroid is ', centroid1)
    s1 = np.sqrt(2)/((1/length)*np.sum((np.sum((x1-centroid1)**2,axis=1))**(0.5)))

    t1_x = -s1 * centroid1[0]
    t1_y = -s1 * centroid1[1]
    # print()
    # print('s1 is', s1)
    
    # sum_x2 = np.sum(x2, axis=0)
    centroid2 = np.mean(x2, axis=0)
    
    s2 = np.sqrt(2)/((1/length)*np.sum((np.sum((x2-centroid1)**2,axis=1))**(0.5)))
    t2_x = -s2 * centroid2[0]
    t2_y = -s2 * centroid2[1]
    
    #%%
    
    T1 = np.array([[s1,0,t1_x],[0,s1,t1_y],[0,0,1]])
    # print('T1 transformation is ', T1)
    T2 = np.array([[s2,0,t2_x],[0,s2,t2_y],[0,0,1]])
    # print('T2 transformation is ', T2)


#%%    
    # turn x1, x2 to homogeneous coordinate
    padding1 = np.ones((x1.shape[0],1))
    padding2 = np.ones((x2.shape[0],1))
    x1_homo = np.append(x1, padding1, axis=1)
    x2_homo = np.append(x2, padding2, axis=1)
    # print('x1 Homogeneous coordinate is ', x1_homo)
    # print('x2 Homogeneous coordinate is ', x2_homo)

    x1_norm_homo = T1 @ x1_homo.T # 3 * N
    x2_norm_homo = T2 @ x2_homo.T # 3 * N
    # print()
    # print('x1 normalized homogeneous coordinate is', x1_norm_homo)
    # print('x2 normalized homogeneous coordinate is', x2_norm_homo)
    # print()
    #Similarity transform 1
    x1_input = x1_norm_homo.T[:, 0:2]
    x2_input = x2_norm_homo.T[:, 0:2]
    # print('x1 input for H computation is ', x1_input)
    # print('x2 input for H computation is ', x2_input)
    # print()
    #Similarity transform 2
    H2to1 = computeH(x1_input, x2_input)
    # print()
    # print('H2to1 is', H2to1)
    #Compute homography
    # print()
    # print('T1^-1 is', np.linalg.inv(T1))
    # print()
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
    #Denormalization
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
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    #定义一个current inliner
    # print()
    # print('input locs1 is ', locs1)
    # print()
    # print('input locs2 is ', locs2)
    # print()
    # print('input locs2 shape is  ', locs2.shape)
    current_inlier = 0
    inliers = np.zeros((locs1.shape[0],1))
    bestH2to1 = np.zeros((3,3))
    for each in range(max_iters):
    # for each in range(1):
    # 选出四个match。来计算homo
        locs = np.append(locs1,locs2, axis=1)
        np.random.shuffle(locs)
        locs_1 = locs[0:4,0:2]
        locs_2 = locs[0:4,2:4]
        # print()
        # print('locs_1 is ', locs_1)
        # print()
        # print('locs_2 is ', locs_2)
        # 计算出Homograph
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
        # print('locs2 is ', locs_2)
        # print()
        # print('The result for locs2 to 1 is', locs2to1)
        locs2to1 = locs2to1.T[:,0:2]
        # print()
        # print('The result for locs2 to 1 is', locs2to1)
        # print()
        # print('locs1 is ', locs_1)
        # 计算偏差
        Bias = locs2to1 - locs1
        # 计算片差距离
        error_distance = np.linalg.norm(Bias, ord=2, axis=1)
        # error_distance = np.sqrt(np.sum(Bias**2,axis=1))

        print()
        print('error distance is', error_distance)
        # 统计inlier
        index_inlier = np.where(error_distance<inlier_tol)
        # print()
        # print('index_inlier is ', index_inlier)
        # print(error_distance)
        # print(index_inlier)
        # 如果index_inlier的个数多余current inlier，则更新
        if index_inlier[0].shape[0]>current_inlier:
             inliers = index_inlier
             bestH2to1 = H2to1
    return bestH2to1, inliers

#%%
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
    composite_img = cv2.warpPerspective(img,)
    return composite_img

