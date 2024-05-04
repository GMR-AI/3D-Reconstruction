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


def get_essential_matrix(K, fundamental_matrices):
    essential_matrix = []
    for i in range(len(fundamental_matrices)):
        essential_matrix.append([])
        for F in fundamental_matrices[i]:
            E = np.dot(K.T, np.dot(F, K))
            u,s,v = np.linalg.svd(E)
            s = [1,1,0]
            essential_matrix[i].append(np.dot(u, np.dot(np.diag(s), v)))

    return essential_matrix
     