import cv2
from ransacImplement import ransac_matrix
import os
import numpy as np

def feature_extraction_set(images):
    sift = cv2.SIFT_create()

    kp, des = [], []
    for im in images:
        kp_tmp, des_tmp = sift.detectAndCompute(im, None) # This assumes the extraction method to be from the CV2 library
        kp.append(kp_tmp)
        des.append(des_tmp)
    return kp, des # Can't turn them into a np array since their shape can be inhomogeneous

def find_matches_and_remove_outliers(keypoints, descriptors, lowes_ratio=0.7):
    # Initialize FLANN matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)    

    matches = []
    n_imgs = len(keypoints)
    for i in range(n_imgs):
        matches.append([])
        for j in range(n_imgs):
            if j <= i: 
                matches[i].append(None)
            else:
                match = []
                m = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                for k in range(len(m)):
                    try:
                        if m[k][0].distance < lowes_ratio*m[k][1].distance:
                            match.append(m[k][0])
                    except:
                        continue
                matches[i].append(match)

    # Remove outliers
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue
            kpts_i = []
            kpts_j = []
            for k in range(len(matches[i][j])):
                kpts_i.append(keypoints[i][matches[i][j][k].queryIdx].pt)
                kpts_j.append(keypoints[j][matches[i][j][k].trainIdx].pt)
            kpts_i = np.int32(kpts_i)
            kpts_j = np.int32(kpts_j)
            F, mask = cv2.findFundamentalMat(kpts_i, kpts_j, cv2.FM_RANSAC, ransacReprojThreshold=3)
            if np.linalg.det(F) > 1e-7: raise ValueError(f"Bad F_mat between images: {i}, {j}. Determinant: {np.linalg.det(F)}")
            matches[i][j] = np.array(matches[i][j])
            if mask is None:
                matches[i][j] = []
                continue
            matches[i][j] = matches[i][j][mask.ravel() == 1]
            matches[i][j] = list(matches[i][j])

            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue

    return matches

def adjacency_matrix(num_imgs, matches):
    num_img_pairs = 0
    num_pairs = 0
    pairs = []
    img_adjacency = np.zeros((num_imgs, num_imgs))
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            num_pairs += 1
            if len(matches[i][j]) > 0:
                num_img_pairs += 1
                pairs.append((i,j))
                img_adjacency[i][j] = 1

    list_of_img_pairs = pairs
    return img_adjacency, list_of_img_pairs