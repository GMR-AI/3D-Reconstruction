import numpy as np
import cv2
from scipy.optimize import least_squares

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


def nonlinear_triangulation(points1, points2, proj_mat1, proj_mat2, X_init):
    num_points = points1.shape[1]
    res = np.ones((4, num_points))
    for i in range(num_points):
        X_init_homog = X_init[:, i] / X_init[3, i]
        result = least_squares(error_and_jacobian, X_init_homog[:3], jac=True, args=(points1[:, i], points2[:, i], proj_mat1, proj_mat2))
        res[:, i] = np.hstack([result.x, 1])
    return res


def linear_triangulation(points1, points2, proj_mat1, proj_mat2):
    num_points = points1.shape[1]
    res = np.ones((4, num_points))
    for i in range(num_points):
        A = np.asarray([
            (points1[0, i] * proj_mat1[2, :] - proj_mat1[0, :]),
            (points1[1, i] * proj_mat1[2, :] - proj_mat1[1, :]),
            (points2[0, i] * proj_mat2[2, :] - proj_mat2[0, :]),
            (points2[1, i] * proj_mat2[2, :] - proj_mat2[1, :])
        ])
        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res