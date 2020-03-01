import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...     p1 = H * p2
    N = p1.shape[1]
    A = np.zeros((2*N, 9))
    for i in range(N):
        x = np.array([p2[0,i], p2[1,i], 1, 0, 0, 0, -p2[0,i]*p1[0,i], -p2[1,i]*p1[0,i], -p1[0,i]])
        y = np.array([0, 0, 0, p2[0,i], p2[1,i], 1, -p2[0,i]*p1[1,i], -p2[1,i]*p1[1,i], -p1[1,i]])
        A[2*i,:] = x
        A[2*i+1,:] = y
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    h = vh[-1,:]
    H2to1 = h.reshape((3,3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    n = 4
    N = matches.shape[0]
    p1 = locs1[matches[:,0], 0:2].T
    p2 = locs2[matches[:,1], 0:2].T
    H = computeH(p1, p2)
    suminlier = 0
    for i in range(num_iter):
        randpts = np.random.randint(N, size = (1,n))
        randp1 = p1[:,randpts].reshape((2,n))
        randp2 = p2[:,randpts].reshape((2,n))
        H = computeH(randp1, randp2)    # p2 = H*p2
        homop2 = np.append(p2, np.array([1]*N).reshape(1,N), axis = 0)
        predp1 = np.dot(H, homop2)
        predp1 /= predp1[-1,:]     # devide by scale factor
        lse = predp1[:2,:] - p1
        lse = np.linalg.norm(lse, axis = 0)
        inlier = np.where(lse<tol, True, False)
        if inlier.sum() > suminlier:
            suminlier = inlier.sum()
            maxinlier = inlier
    p1 = p1[:,maxinlier]
    p2 = p2[:,maxinlier]
    bestH = computeH(p1, p2)
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

