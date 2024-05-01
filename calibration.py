import numpy as np
import cv2
import glob

def calibrate_camera(chessboard_size, images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save intrinsic matrix (mtx) and distortion coefficients (dist) to text files
    np.savetxt('intrinsic_matrix.txt', mtx, fmt='%f')
    np.savetxt('distortion_coefficients.txt', dist, fmt='%f')

    return ret, mtx, dist, rvecs, tvecs


def estimate_pose(pts1, pts2, K):
    # Normalize points
    pts1_normalized = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
    pts2_normalized = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)

    # Find essential matrix
    E, mask = cv2.findEssentialMat(pts1_normalized, pts2_normalized, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    points, R, t, mask = cv2.recoverPose(E, pts1_normalized, pts2_normalized)

    return R, t

def essential_to_projection(essential_matrix, proj_mat):
    u, s, v = np.linalg.svd(essential_matrix)

    if np.linalg.det(np.dot(u, v)) < 0:
        v = -v

    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    p2s = [np.vstack((np.dot(u, np.dot(w, v)).T, u[:, 2])).T,
           np.vstack((np.dot(u, np.dot(w, v)).T, -u[:, 2])).T,
           np.vstack((np.dot(u, np.dot(w.T, v)).T, u[:, 2])).T,
           np.vstack((np.dot(u, np.dot(w.T, V)).T, -u[:, 2])).T]

    return p2s
    pass