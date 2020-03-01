import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    cornors = np.array([[0,0,im2.shape[1]-1,im2.shape[1]-1], [0,im2.shape[0]-1,0,im2.shape[0]-1]])
    cornors = np.append(cornors, np.array([1]*4).reshape(1,4), axis = 0)
    trans_cornors = np.dot(H2to1, cornors)
    trans_cornors /= trans_cornors[-1,:]
    trans_cornors = trans_cornors[:2,:]
    coord_max = trans_cornors.max(axis = 1)
    im_size = tuple(np.ceil(coord_max).astype(int))
    # coord_min = trans_cornors.min(axis = 1)
    # im_size = tuple(np.ceil(np.maximum(coord_max, coord_max - coord_min)).astype(int))

    im_trans = cv2.warpPerspective(im2, H2to1, im_size)
    # cv2.imwrite('../results/q6_1.jpg', im_trans)
    # np.save('../results/q6_1.npy', H2to1)
    # im_trans[:im1.shape[0], :im1.shape[1], :] = im1   # no blend

    im1_broaden = np.zeros_like(im_trans)
    im1_broaden[:im1.shape[0], :im1.shape[1], :] = im1
    mask_overlap = np.logical_and(im1_broaden, im_trans)
    pano_im = np.where(mask_overlap == True, np.maximum(im_trans, im1_broaden), im_trans + im1_broaden) # maxblend
    # pano_im = np.where(mask_overlap == True, (im_trans + im1_broaden)/2/255, (im_trans + im1_broaden)/255)    # average but weird
    # cv2.imwrite('../results/pano_maxblend.jpg', pano_im)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    cornors = np.array([[0,0,im2.shape[1]-1,im2.shape[1]-1], [0,im2.shape[0]-1,0,im2.shape[0]-1]])
    cornors = np.append(cornors, np.array([1]*4).reshape(1,4), axis = 0)
    trans_cornors = np.dot(H2to1, cornors)
    trans_cornors /= trans_cornors[-1,:]
    trans_cornors = trans_cornors[:2,:]
    coord_max = trans_cornors.max(axis = 1)
    coord_min = trans_cornors.min(axis = 1)
    im_size = tuple(np.ceil(np.maximum(coord_max, coord_max - coord_min)).astype(int))
    tx = np.absolute(np.minimum(0,coord_min[0]))
    ty = np.absolute(np.minimum(0,coord_min[1]))
    M = np.array([[1,0,tx], [0,1,ty], [0,0,1]])
    H = np.dot(M, H2to1)

    im2_trans = cv2.warpPerspective(im2, H, im_size)
    im1_trans = cv2.warpPerspective(im1, M, im_size)
    mask_overlap = np.logical_and(im1_trans, im2_trans)
    pano_im = np.where(mask_overlap == True, np.maximum(im1_trans, im2_trans), im1_trans + im2_trans) # maxblend
    # cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    return pano_im


def generatePanorama(im1, im2):
    '''
    Returns a panorama of im1 and im2 without cliping.
    ''' 
    ######################################
    # TO DO ...
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    
    # pano_im = imageStitching(im1, im2, H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    pano_im = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()