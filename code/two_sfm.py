from typing import Literal

import cv2
import numpy as np
import features as ft
import triangulation as tr
import output_classes as oc

_METHODS = ['cv2', 'custom']


def starting_img_set(adj_matrix: np.ndarray, img_db: dict[int, oc.c_Image], matches: list[list], K: np.ndarray, top_x_perc: float = 0.2):
    """
    Find the best two images to start with given that they have enough matches and a big rotation between one and the other.

    adj_matrix: NxN adjacency matrix that connects each image with those they have some match with.
    kp: list of tuples of keypoints of all images.
    matches: list matrix of matches between all images.
    K: 3x3 matrix with camera intrinsics.
    top_x_perc: percentage threshold to consider image pairs. e.g 0.2 means top 80%
    """

    n_matches = [len(matches[i][j]) for j in range(adj_matrix.shape[1]) for i in range(adj_matrix.shape[0]) if adj_matrix[i][j] == 1]
    n_matches = sorted(n_matches, reverse=True)
    min_match_idx = int(len(n_matches) * top_x_perc)
    min_matches = n_matches[min_match_idx]
    best_R = 0
    best_pair = None
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                if len(matches[i][j]) > min_matches:
                    kpts_i, kpts_j = ft.matching_keypoints(img_db, matches, i+1, j+1)
                    kpts_i = np.expand_dims(kpts_i.T, axis=1)
                    kpts_j = np.expand_dims(kpts_j.T, axis=1)
                    E, _ = cv2.findEssentialMat(kpts_i, kpts_j, K, cv2.FM_RANSAC, 0.999, 1.0)
                    points, R1, t1, mask = cv2.recoverPose(E, kpts_i, kpts_j, K)
                    rvec, _ = cv2.Rodrigues(R1)
                    rot_angle = abs(rvec[0]) +abs(rvec[1]) + abs(rvec[2])# sum rotation angles for each dimension
                    if (rot_angle > best_R or best_pair == None) and points == len(kpts_i): #Ensure recoverPose worked.
                        best_R = rot_angle
                        best_pair = (i,j)
    
    return best_pair


def get_camera_pose(img_db: dict[int, oc.c_Image], matches: list[list], K: np.ndarray, img_idx1: int, img_idx2: int, method: Literal['cv2', 'custom'] = 'cv2'):
    """
    Get the camera pose for image 2 relative to image 1.

    kp: list of tuples of keypoints of all images.
    matches: list matrix of matches between all images.
    K: 3x3 matrix with camera intrinsics.
    img_idx1: index of image 1.
    img_idx2: index of image 2.
    method: defines if it uses the cv2 implemented way or a custom method.
    """

    if method not in _METHODS: raise ValueError(f'Method {method} not valid')
    
    pts2d1, pts2d2 = ft.matching_keypoints(img_db, matches, img_idx1+1, img_idx2+1)
    R1 = np.eye(3)
    t1 = np.expand_dims(np.zeros(3), axis=1)

    if method == 'cv2':
        pts2d1 = np.expand_dims(pts2d1.T, axis=1)
        pts2d2 = np.expand_dims(pts2d2.T, axis=1)
        E, _ = cv2.findEssentialMat(pts2d1, pts2d2, K, cv2.FM_RANSAC, 0.999, 1.0)
        _, R2, t2, _ = cv2.recoverPose(E, pts2d1, pts2d2, K)
        pts3d = tr.triangulate(K, R1, t1, R2, t2, pts2d1, pts2d2, method='cv2')
    
    if method == 'custom':
        F = fundamental_matrix(pts2d1.T, pts2d2.T)
        E = essential_from_fundamental(K, F)
        R2s, t2s = pose_from_essential(E)
        R2, t2, pts3d = double_disambiguation(K, R1, t1, R2s, t2s, pts2d1, pts2d2)
    
    return R1, t1, R2, t2, pts3d


def fundamental_matrix(pts2d1: np.ndarray, pts2d2: np.ndarray):
    """
    Calculate the fundamental matrix of the camera 2 using the eight point algorithm.

    pts2d1: array of points 2d of the first image matching the points 2d of the second image. Must be 2xN.
    pts2d2: array of points 2d of the second image matching the points 2d of the first image. Must be 2xN.
    """
    pts1_homo = np.vstack((pts2d1, np.ones(pts2d1.shape[1]))).T
    pts2_homo = np.vstack((pts2d2, np.ones(pts2d2.shape[1]))).T

    # Normalization
    T = normalize(pts1_homo)
    T_prime = normalize(pts2_homo)


    pts1_homo = (T @ pts1_homo.T).T
    pts2_homo = (T_prime @ pts2_homo.T).T

    # x2.T*F*x1=0
    # A*f=0, f is F flattened into a 1D array

    # Create A
    A = np.zeros((pts2d1.shape[0], 9))
    for i in range(pts2d2.shape[0]):
        A[i] = np.array([
            pts1_homo[i,0]*pts2_homo[i,0], pts1_homo[i,1]*pts2_homo[i,0], pts1_homo[i,2]*pts2_homo[i,0],
            pts1_homo[i,0]*pts2_homo[i,1], pts1_homo[i,1]*pts2_homo[i,1], pts1_homo[i,2]*pts2_homo[i,1],
            pts1_homo[i,0]*pts2_homo[i,2], pts1_homo[i,1]*pts2_homo[i,2], pts1_homo[i,2]*pts2_homo[i,2]
            ])
    
    # Solve Af=0 using svd
    U,S,Vt = np.linalg.svd(A)
    F = Vt[-1,:].reshape((3,3))

    # Enforce rank2 constraint
    U,S,Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    F = T_prime.T @ F @ T
    return F


def essential_from_fundamental(K: np.ndarray, F: np.ndarray):
    """
    Computes the essential matrix from the fundamental matrix.

    K: 3x3 matrix with camera intrinsics.
    F: 3x3 fundamental matrix
    """
    return K.T @ F @ K


def pose_from_essential(E: np.ndarray):
    """
    Gets the 4 possible camera poses from the essential matrix.
    
    E: 3x3 essential matrix
    """
    U,_,Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    Rs = [U @ W @ Vt, U @ W.T @ Vt]
    for i in range(len(Rs)):
        if np.linalg.det(Rs[i]) < 0:
            Rs[i] = Rs[i] * -1
    
    ts = [U[:, 2, np.newaxis], -U[:, 2, np.newaxis]]

    return np.array(Rs), np.array(ts)


def double_disambiguation(K: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2s: np.ndarray, t2s: np.ndarray, pts2d1: np.ndarray, pts2d2: np.ndarray):
    """
    Define the most possible pose for camera 2.

    K: 3x3 matrix with camera intrinsics.
    R1: 3x3 rotation matrix of camera 1.
    t1: 3x1 position vector of camera 1.
    R2s: 4x3x3 rotation matrix possibilities of camera 1.
    t2s: 4x3x1 position vector possibilities of camera 1.
    pts2d1: 2xN 2D points of image 1 corresponding to the points of image 2.
    pts2d2: 2xN 2D points of image 2 corresponding to the points of image 1.
    """

    # Triangulate points
    pts3d_possible = []
    for R2 in R2s:
        pts3d_possible.append([])
        for t2 in t2s:
            pts3d_possible[-1].append(tr.triangulate_and_reproject(K, R1, t1, R2, t2, pts2d1, pts2d2, method='custom'))
    
    pts3d_possible = np.array(pts3d_possible)

    # Disambiguate
    max_positive_z = 0
    min_error = np.finfo('float').max
    best_R = None
    best_t = None
    best_pts3d = None

    for i in range(R2s.shape[0]):
        for j in range(t2s.shape[0]):
            num_positive_z = np.sum(pts3d_possible[i][2, :] > 0)
            repr_err1, _, _ = tr.reprojection_error(K, R1, t1, pts2d1, pts3d_possible[i+i*j])
            repr_err2, _, _ = tr.reprojection_error(K, R2s[i], t2s[j], pts2d2, pts3d_possible[i+i*j])
            repr_err = repr_err1 + repr_err2

            if num_positive_z >= max_positive_z and repr_err < min_error:
                max_positive_z = num_positive_z
                min_error = repr_err
                best_R = R2s[i]
                best_t = t2s[j]
                best_pts3d = pts3d_possible[i+i*j]
    
    return best_R, best_t, best_pts3d


def normalize(pts: np.ndarray):
    """
    Normalize points for the eight point algorithm

    pts: 2D points to normalize. Must be Nx2
    """

    x_mean = np.mean(pts[:, 0])
    y_mean = np.mean(pts[:, 1])
    sigma = np.mean(np.sqrt((pts[:, 0] - x_mean) ** 2 + (pts[:, 1] - y_mean) ** 2))
    M = np.sqrt(2) / sigma
    T = np.array([
        [M, 0, -M * x_mean],
        [0, M, -M * y_mean],
        [0, 0, 1]
    ])
    return T


if __name__ == '__main__':
    fundamental_matrix(np.ndarray(0), np.ndarray(0))