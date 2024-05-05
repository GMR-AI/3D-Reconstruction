import numpy as np
import cv2
import triangulation
import glob

def calibrate_camera(chessboard_size, images):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

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


def ExtractCameraPose(E, K):
    # Descomponer la matriz esencial en R y t
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Cuatro posibles combinaciones de R y t
    R = [np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(W.T, Vt)), np.dot(U, np.dot(W.T, Vt))]
    t = [U[:, 2], U[:, 2], -U[:, 2], -U[:, 2]]

    # Asegurarse de que la determinante de R es positiva
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            t[i] = -t[i]
    
    proj_mats = [K @ np.hstack((R[i], np.transpose([t[i]]))) for i in range(4)]

    return proj_mats, R, t


def ambiguity_solver(proj_mat1, proj_mats2, point1, point2):
    for i, proj_mat2 in enumerate(proj_mats2):
        a1 = triangulation.linear_triangulation(point1, point2, proj_mat1, proj_mat2)
        pm2_homo = np.linalg.inv(np.vstack([proj_mat2, [0, 0, 0, 1]]))
        a2 = np.dot(pm2_homo[:3, :4], a1)
        if a1[2] > 0 and a2[2] > 0:
            return proj_mat2

    return None


def disambiguate_camera_pose(Rs, ts, points1, points2, K):
    max_count = 0
    for i in range(4):
        R = Rs[i]
        t = ts[i]
        P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(K, np.hstack((R, t.reshape(-1, 1))))
        initial_points = triangulation.linear_triangulation(points1, points2, P1, P2)
        points3D = triangulation.nonlinear_triangulation(points1, points2, P1, P2, initial_points)
        count = np.sum((np.dot(P1[2], points3D.T) > 0) & (np.dot(P2[2], points3D.T) > 0))
        if count > max_count:
            max_count = count
            best_R = R
            best_t = t
    return best_R, best_t

def disambiguate(pts1, pts2, P1, P2s):
    P1 = np.array(P1)
    P2s = np.array(P2s)
    best_error = np.finfo('float').max
    for i in range(P2s.shape[0]):
        P2_tmp = P2s[i]
        X, err = triangulation.linear_triangulation2(pts1, pts2, P1, P2_tmp)
        print(X.shape)
        print(err < best_error)
        print(np.all(X[-1, :] >= 0))
        if err < best_error and np.all(X[-1, :] >= 0):
            best_error = err
            P2 = P2_tmp
    return P2