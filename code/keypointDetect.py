import numpy as np
import cv2
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    for i in DoG_levels:
        DoG_pyramid.append(gaussian_pyramid[:,:,i+1] - gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    ddepth = -1
    dxx = cv2.Sobel(DoG_pyramid, ddepth, 2, 0)
    dyy = cv2.Sobel(DoG_pyramid, ddepth, 0, 2)
    dxy = cv2.Sobel(DoG_pyramid, ddepth, 1, 1)
    principal_curvature = (dxx+dyy)**2 / (dxx*dyy - dxy*dxy)
    # displayPyramid(dxx)
    # displayPyramid(dyy)
    # displayPyramid(dxy)
    # print(np.max(principal_curvature[:,:,0]), np.min(principal_curvature[:,:,0]))
    # print(principal_curvature[:,:,0])
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    
    mask = []
    # shift to the right pad tuple (up, down), (left, right), (above, behind))
    mask.append(np.pad(DoG_pyramid, ((0,0),(1,0),(0,0)), mode = 'constant', constant_values = 0)[:,:-1,:])
    # shift to the left
    mask.append(np.pad(DoG_pyramid, ((0,0),(0,1),(0,0)), mode = 'constant', constant_values = 0)[:,1:,:])
    # shift up
    mask.append(np.pad(DoG_pyramid, ((0,1),(0,0),(0,0)), mode = 'constant', constant_values = 0)[1:,:,:])
    # shift down
    mask.append(np.pad(DoG_pyramid, ((1,0),(0,0),(0,0)), mode = 'constant', constant_values = 0)[:-1,:,:])
    # shift upper right
    mask.append(np.pad(DoG_pyramid, ((0,1),(1,0),(0,0)), mode = 'constant', constant_values = 0)[1:,:-1,:])
    # shift upper left
    mask.append(np.pad(DoG_pyramid, ((0,1),(0,1),(0,0)), mode = 'constant', constant_values = 0)[1:,1:,:])
    # shift lower right
    mask.append(np.pad(DoG_pyramid, ((1,0),(1,0),(0,0)), mode = 'constant', constant_values = 0)[:-1,:-1,:])
    # shift lower left
    mask.append(np.pad(DoG_pyramid, ((1,0),(0,1),(0,0)), mode = 'constant', constant_values = 0)[:-1,1:,:])
    mask = np.stack(mask, axis=-1)
    # print("Mask shape: ", mask.shape)

    locsExtrema = []
    aug_pyramid = np.pad(DoG_pyramid, ((0,0),(0,0),(1,1)), mode = 'constant', constant_values = 0)
    for i in DoG_levels:
        layer_mask = [aug_pyramid[:,:,i], aug_pyramid[:,:,i+2]]
        for j in range(mask.shape[-1]):
            layer_mask.append(mask[:,:,i,j])
        layer_mask = np.stack(layer_mask, axis = -1)

        layer_max = np.max(layer_mask, axis = -1)
        layer_min = np.min(layer_mask, axis = -1)
        layer_max = np.where(aug_pyramid[:,:,i+1] > layer_max, True, False)
        layer_min = np.where(aug_pyramid[:,:,i+1] < layer_min, True, False)
        layer_extreme = np.logical_or(layer_max, layer_min)
        locsExtrema.append(layer_extreme)

    locsExtrema = np.stack(locsExtrema, axis = -1)
    locsContrast = np.where(np.absolute(DoG_pyramid)>th_contrast, True, False)
    locsNonEdges = np.where(principal_curvature < th_r, True, False)
    locsDoGmap = np.logical_and(locsExtrema, locsContrast)
    locsDoGmap = np.logical_and(locsDoGmap, locsNonEdges)
    locsDoG = np.argwhere(locsDoGmap == True)  
    # print("# Keypoints: ", locsDoG.shape[0])
    return locsDoG

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0 = sigma0, k = k, levels = levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels = levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    implot = plt.imshow(im)
    plt.scatter(locsDoG[:,1], locsDoG[:,0], c = 'g')
    plt.show()


'''Buggy's here'''
# def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
#         th_contrast=0.03, th_r=12):
#     '''
#     Returns local extrema points in both scale and space using the DoGPyramid

#     INPUTS
#         DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
#         DoG_levels  - The levels of the pyramid where the blur at each level is
#                       outputs
#         principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
#                       curvature ratio R
#         th_contrast - remove any point that is a local extremum but does not have a
#                       DoG response magnitude above this threshold
#         th_r        - remove any edge-like points that have too large a principal
#                       curvature ratio
#      OUTPUTS
#         locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
#                scale and space, and also satisfies the two thresholds.
#     '''
#     locsDoG = None
#     ##############
#     #  TO DO ...
#     # Compute locsDoG here
    
#     mask = []
#     # shift to the right pad tuple (up, down), (left, right), (above, behind))
#     mask.append(np.pad(DoG_pyramid, ((0,0),(1,0),(0,0)), mode = 'constant', constant_values = 0)[:,:-1,:])
#     # shift to the left
#     mask.append(np.pad(DoG_pyramid, ((0,0),(0,1),(0,0)), mode = 'constant', constant_values = 0)[:,1:,:])
#     # shift up
#     mask.append(np.pad(DoG_pyramid, ((0,1),(0,0),(0,0)), mode = 'constant', constant_values = 0)[1:,:,:])
#     # shift down
#     mask.append(np.pad(DoG_pyramid, ((1,0),(0,0),(0,0)), mode = 'constant', constant_values = 0)[:-1,:,:])
#     # shift upper right
#     mask.append(np.pad(DoG_pyramid, ((0,1),(1,0),(0,0)), mode = 'constant', constant_values = 0)[1:,:-1,:])
#     # shift upper left
#     mask.append(np.pad(DoG_pyramid, ((0,1),(0,1),(0,0)), mode = 'constant', constant_values = 0)[1:,1:,:])
#     # shift lower right
#     mask.append(np.pad(DoG_pyramid, ((1,0),(1,0),(0,0)), mode = 'constant', constant_values = 0)[:-1,:-1,:])
#     # shift lower left
#     mask.append(np.pad(DoG_pyramid, ((1,0),(0,1),(0,0)), mode = 'constant', constant_values = 0)[:-1,1:,:])
#     mask = np.stack(mask, axis=-1)
#     print("Mask shape: ", mask.shape)

#     locsDoGmap = []
#     aug_pyramid = np.pad(DoG_pyramid, ((0,0),(0,0),(1,1)), mode = 'constant', constant_values = 0)
#     for i in DoG_levels:
#         # apply th_contrast threshold on DoG
#         # layer_mask = [np.where(np.absolute(aug_pyramid[:,:,i+1])>th_contrast, aug_pyramid[:,:,i+1], 0),
#         #                 aug_pyramid[:,:,i], aug_pyramid[:,:,i+2]]
#         layer_mask = [aug_pyramid[:,:,i+1], aug_pyramid[:,:,i], aug_pyramid[:,:,i+2]]
#         for j in range(mask.shape[-1]):
#             layer_mask.append(mask[:,:,i,j])
#         layer_mask = np.stack(layer_mask, axis = -1)
#         print("Layer mask shape: ", layer_mask.shape)

#         layer_max = np.argmax(layer_mask, axis = -1)
#         layer_min = np.argmin(layer_mask, axis = -1)
#         layer_max = np.where(layer_max == 0, 1, 0)
#         layer_min = np.where(layer_min == 0, 1, 0)
#         layer_extreme = np.logical_or(layer_max, layer_min)
#         locsDoGmap.append(layer_extreme)
#         # print(layer_max)
#         # print(layer_min)        
#         # print(layer_extreme)
#         # break
#     locsDoGmap = np.stack(locsDoGmap, axis = -1)
#     locsDoGmap = np.where(np.absolute(locsDoGmap)>th_contrast, True, False)
#     edge_removed = np.where(principal_curvature < th_r, True, False)
#     locsDoGmap = np.logical_and(locsDoGmap, edge_removed)
#     locsDoG = np.argwhere(locsDoGmap == True)        
#     # print(locsDoGmap.shape)
#     print("# points: ", locsDoG.shape[0])
#     return locsDoG
