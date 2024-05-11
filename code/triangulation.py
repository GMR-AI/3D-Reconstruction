from typing import Literal

import cv2
import numpy as np

_METHODS = ['cv2', 'custom']

def triangulate(K: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray, pts2d1: np.ndarray, pts2d2: np.ndarray, method: Literal['cv2', 'custom'] = 'cv2'):
    """
    Calculate the linear triangulation between all matching points.

    K: 3x3 matrix with camera intrinsics.
    R1: 3x3 rotation matrix of camera 1.
    t1: 3x1 position vector of camera 1.
    R2s: 3x3 rotation matrix of camera 1.
    t2s: 3x1 position vector of camera 1.
    pts2d1: 2xN 2D points of image 1 corresponding to the points of image 2.
    pts2d2: 2xN 2D points of image 2 corresponding to the points of image 1.
    method: defines if it uses the cv2 implemented way or a custom method.

    return:
    pts3d
    """

    if method not in _METHODS: raise ValueError(f'Method {method} not valid')

    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])

    if method == 'cv2':
        pts3d_homo = cv2.triangulatePoints(P1, P2, pts2d1, pts2d2)
        pts3d = pts3d_homo[:3] / pts3d_homo[-1]
    
    if method == 'custom':
        pts3d = linear_triangulation(P1, P2, pts2d1, pts2d2)
    
    return pts3d


def triangulate_and_reproject(K: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray, pts2d1: np.ndarray, pts2d2: np.ndarray, reproject: bool = True, method: Literal['cv2', 'custom'] = 'cv2'):
    """
    Calculate the linear triangulation between all matching points and their reprojection error if necessary.

    K: 3x3 matrix with camera intrinsics.
    R1: 3x3 rotation matrix of camera 1.
    t1: 3x1 position vector of camera 1.
    R2s: 3x3 rotation matrix of camera 1.
    t2s: 3x1 position vector of camera 1.
    pts2d1: 2xN 2D points of image 1 corresponding to the points of image 2.
    pts2d2: 2xN 2D points of image 2 corresponding to the points of image 1.
    reproject: If true, then it also calculates the reprojection error and returns it
    method: defines if it uses the cv2 implemented way or a custom method.

    return:
    pts3d
    """

    pts3d = triangulate(K, R1, t1, R2, t2, pts2d1, pts2d2)

    if reproject:
        _, avg_error1, _ = reprojection_error(K, R1, t1, pts2d1, pts3d)
        _, avg_error2, _ = reprojection_error(K, R2, t2, pts2d2, pts3d)

        return pts3d, avg_error1, avg_error2

    return pts3d



def linear_triangulation(P1: np.ndarray, P2: np.ndarray, pts2d1: np.ndarray, pts2d2: np.ndarray):
    """
    Custom linear triangulation implementation between all matching points.

    P1: 3x4 projection matrix of camera 1.
    P2: 3x4 projection matrix of camera 2.
    pts2d1: 2xN 2D points of image 1 corresponding to the points of image 2.
    pts2d2: 2xN 2D points of image 2 corresponding to the points of image 1.

    return:
    pts3d
    """

    # First, set all points to homogeneous
    pts1_homo = np.vstack((pts2d1, np.ones(pts2d1.shape[1])))
    pts2_homo = np.vstack((pts2d2, np.ones(pts2d2.shape[1])))

    # Solve using svd
    pts3d = np.zeros((3, pts2d1.shape[1]))
    for i in range(pts2d2.shape[1]):
        A = np.array([pts1_homo[1,i]*P1[2,:] - P1[1,:],
                      P1[0,:] - pts1_homo[0,i]*P1[2,:],
                      pts2_homo[1,i]*P2[2,:] - P2[1,:],
                      P2[0,:] - pts2_homo[0,i]*P2[2,:]])
        ATA = A.T @ A
        _, _, Vt = np.linalg.svd(ATA)
        if Vt[-1, -1] != 0:
            pts3d[:, i] = Vt[-1, :3]/Vt[-1, -1]
    
    return pts3d


def reprojection(P: np.ndarray, pts3d: np.ndarray):
    """
    Calculate the reprojection of a set 3D points to an image.

    P: 3x4 projection matrix of the camera.
    pts3d: 3xN 3D points to reproject onto the image.

    return:
    pts2d
    """
    pts3d_homo = np.vstack([pts3d, np.ones(pts3d.shape[1])])
    pts2d_homo = P @ pts3d_homo
    pts2d = pts2d_homo[:2]/pts2d_homo[-1]
    return pts2d


def reprojection_error(K: np.ndarray, R: np.ndarray, t: np.ndarray, pts2d: np.ndarray, pts3d: np.ndarray, rep_threshold = 5):
    """
    Calculate the reprojection error of the 3d points to the image.

    K: 3x3 matrix with camera intrinsics.
    R: 3x3 rotation matrix of the camera.
    t: 3x1 position vector of the camera.
    pts2d: 2xN 2D points of the image corresponding to the 3D points.
    pts3d: 3xN 3D points corresponding to the 2D points of the image.
    rep_threshold: reprojection error threshold to consider the inliers

    return:
    total_error, avg_error, perc_inliers
    """

    P = K @ np.hstack([R, t])
    pts2d_repr = reprojection(P, pts3d)

    error_list = np.sum(np.square(pts2d_repr - pts2d), axis=0)
    total_error = np.sum(error_list)
    avg_error = total_error/len(error_list)

    inliers = error_list <= rep_threshold
    perc_inliers = np.sum(inliers) / len(inliers)
    
    return total_error, avg_error, perc_inliers


if __name__ == '__main__':
    pts3d = np.array([3, 4, 5])
    pts3d = np.expand_dims(pts3d, axis=1)
    P = np.hstack([np.eye(3), np.zeros((3,1))])

    reprojection(P, pts3d)