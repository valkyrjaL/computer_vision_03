import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import BRIEF


if __name__ == '__main__':
    # load test pattern for Brief
    test_pattern_file = '../results/testPattern.npy'
    if os.path.isfile(test_pattern_file):
        # load from file if exists
        compareX, compareY = np.load(test_pattern_file)
    else:
        # produce and save patterns if not exist
        compareX, compareY = BRIEF.makeTestPattern()
        if not os.path.isdir('../results'):
            os.mkdir('../results')
        np.save(test_pattern_file, [compareX, compareY])
    
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    height, width = im1.shape[:2]
    image_center = (width / 2, height / 2)
    locs1, desc1 = BRIEF.briefLite(im1)
    hist = []
    angs = np.arange(0, 360, 10)
    for angle in angs:        
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
        im2 = cv2.warpAffine(im1, rotation_mat, (width, height))
        locs2, desc2 = BRIEF.briefLite(im2)
        matches = BRIEF.briefMatch(desc1, desc2)
        hist.append(matches.shape[0])
        # BRIEF.plotMatches(im1,im2,matches,locs1,locs2)
    plt.bar(angs, hist)
    plt.xlabel("rotation degree")
    plt.ylabel("number of matches")
    plt.show()
