import numpy as np
import cv2

def ransac_matrix(matches, keypoints):
    fundamental_matrices = []
    for i in range(len(matches)):  # avoid accessing out-of-range index
        fundamental_matrices.append([])
        for j in range(len(matches[i])):
            idx2 = (i + (j - int(len(matches[i])/2))) % len(keypoints)
            pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i][j]]).reshape(-1,1,2)
            pts2 = np.float32([keypoints[idx2][m.trainIdx].pt for m in matches[i][j]]).reshape(-1,1,2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            fundamental_matrices[i].append(F)
    return fundamental_matrices