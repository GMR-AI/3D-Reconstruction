import numpy as np
from scipy.optimize import leastsq

def project(points, P):
    points_proj = P @ np.vstack((points.T, np.ones((1, points.shape[0]))))
    points_proj /= points_proj[2, :]
    return points_proj[:2, :].T

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    cameras = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

    points_proj = np.empty_like(points_2d)

    for i in range(n_cameras):
        points_proj[camera_indices == i] = project(points_3d[point_indices[camera_indices == i]], K @ cameras[i])

    return (points_proj - points_2d).ravel()

def bundle_adjustment(points_2d, camera_indices, point_indices, initial_cameras, initial_points, K):
    n_cameras = initial_cameras.shape[0]
    n_points = initial_points.shape[0]

    x0 = np.hstack((initial_cameras.ravel(), initial_points.ravel()))

    f0 = reprojection_error(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, K)

    A = np.identity(len(x0))
    A = A * 1e-3

    res = leastsq(reprojection_error, x0, args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K), Dfun=None, full_output=True, ftol=1e-5, xtol=1e-5, maxfev=10000, epsfcn=1e-6, factor=10)

    return res[0]