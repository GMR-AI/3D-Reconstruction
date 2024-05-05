import numpy as np
import cv2
from scipy.optimize import least_squares


def project(points, P):
    points = np.dot(P, points.T).T
    points[:, :2] /= points[:, 2, np.newaxis]
    return points[:, :2]

def reprojection_error(params, points1, points2, P1, P2):
    points3D = params.reshape(-1, 3)
    points3D_homogeneous = np.hstack([points3D, np.ones((points3D.shape[0], 1))])
    projected_points1 = project(points3D_homogeneous, P1)
    projected_points2 = project(points3D_homogeneous, P2)
    return np.hstack([(points1 - projected_points1).ravel(), (points2 - projected_points2).ravel()])

def nonlinear_triangulation(points1, points2, P1, P2, initial_points):
    initial_params = initial_points.ravel()
    res = least_squares(reprojection_error, initial_params, method='lm', args=(points1, points2, P1, P2))
    return res.x.reshape(-1, 3)


def error_and_jacobian(x, points1, points2, proj_mat1, proj_mat2):
    X = np.hstack([x, 1])
    x1_proj = np.dot(proj_mat1, X)
    x2_proj = np.dot(proj_mat2, X)
    x1_proj /= x1_proj[2]
    x2_proj /= x2_proj[2]
    error = np.hstack([points1 - x1_proj[:2], points2 - x2_proj[:2]])
    jacobian = np.zeros((4, 3))
    jacobian[:2, :] = proj_mat1[2, :3] - (proj_mat1[:2, :3].T * x1_proj[2]).T
    jacobian[2:, :] = proj_mat2[2, :3] - (proj_mat2[:2, :3].T * x2_proj[2]).T
    return error, jacobian


def linear_triangulation(points1, points2, proj_mat1, proj_mat2):
    num_points = np.shape(points1)[0]
    res = np.ones((4, num_points))
    for i in range(num_points):
        A = np.squeeze(np.asarray([
            [points1[i][0] * proj_mat1[2][:] - proj_mat1[0][:]],
            [points1[i][1] * proj_mat1[2][:] - proj_mat1[1][:]],
            [points2[i][0] * proj_mat2[2][:] - proj_mat2[0][:]],
            [points2[i][1] * proj_mat2[2][:] - proj_mat2[1][:]]
        ]))
        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res

def linear_triangulation2(pts1, pts2, P1, P2):
    # Calculate 3D point
    pts4D = np.zeros((4, pts1.shape[1]))
    pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2, pts4D)
    pts4D /= pts4D[3, :]

    # Calculate reprojection error
    # First, get reprojections
    pts1_reproj = P1 @ pts4D
    pts2_reproj = P2 @ pts4D

    pts1_reproj /= pts1_reproj[-1, :]
    pts2_reproj /= pts2_reproj[-1, :]

    # Second, homogenize every point to compare with the reprojection
    # pts1 (2xN)
    pts1_homo = np.concatenate((pts1, np.ones((1, pts1.shape[1]))), axis=0)
    pts2_homo = np.concatenate((pts2, np.ones((1, pts2.shape[1]))), axis=0)

    err1 = np.sum(np.square(pts1_reproj - pts1_homo))
    err2 = np.sum(np.square(pts2_reproj - pts2_homo))

    err = err1 + err2
    return pts4D[:3, :], err