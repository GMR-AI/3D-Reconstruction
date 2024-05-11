import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from colmap_functions import rotmat2qvec, qvec2rotmat

import output_classes as oc


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A


def project(points, camera_params, K):
    points_proj = []

    for idx in range(len(camera_params)):  # idx applies to both points and cam_params, they are = length vectors
        R = camera_params[idx][:9].reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)
        t = camera_params[idx][9:]
        pt = points[idx]
        pt = np.expand_dims(pt, axis=0)
        pt, _ = cv2.projectPoints(pt, rvec, t, K, distCoeffs=np.array([]))
        pt = np.squeeze(np.array(pt))
        points_proj.append(pt)

    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()


def bundleAd(pts3D_db: dict[int, oc.c_Point3D], img_db: dict[int, oc.c_Image], processed_imgs, K, ftol):
    R_mats = {id-1: qvec2rotmat(im.qvec) for id, im in img_db.items() if isinstance(im.qvec, np.ndarray) }
    t_vecs = {id-1: im.tvec for id, im in img_db.items() if isinstance(im.qvec, np.ndarray)}

    point_indices = []
    points_2d = []
    camera_indices = []
    points_3d = []
    camera_params = []
    BA_cam_idxs = {} # maps from true cam indices to 'normalized' (i.e 11, 23, 31 maps to -> 0, 1, 2)
    cam_count = 0

    for r in processed_imgs:
        BA_cam_idxs[r] = cam_count
        camera_params.append(np.hstack((R_mats[r].ravel(), t_vecs[r].ravel())))
        cam_count += 1

    for pt3d_idx in range(len(pts3D_db)):
        points_3d.append(pts3D_db[pt3d_idx].xyz)
        for cam_idx, kpt_idx in zip(pts3D_db[pt3d_idx].image_ids, pts3D_db[pt3d_idx].point2D_idxs):
            if cam_idx - 1 not in processed_imgs: continue
            point_indices.append(pt3d_idx)
            camera_indices.append(BA_cam_idxs[cam_idx - 1])  # append normalized cam idx
            points_2d.append(img_db[cam_idx].xys[kpt_idx])

    if len(points_3d[0]) == 3: points_3d = np.expand_dims(points_3d, axis=0)

    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    camera_indices = np.array(camera_indices)
    points_3d = np.squeeze(points_3d)
    camera_params = np.array(camera_params)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', loss='linear', ftol=ftol, xtol=1e-12, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))

    adjusted_camera_params = res.x[:n_cameras * 12].reshape(n_cameras, 12)
    adjusted_points_3d = res.x[n_cameras * 12:].reshape(n_points, 3)
    adjusted_R_mats = {}
    adjusted_t_vecs = {}
    for true_idx, norm_idx in BA_cam_idxs.items():
        adjusted_R_mats[true_idx] = adjusted_camera_params[norm_idx][:9].reshape(3,3)
        adjusted_t_vecs[true_idx] = adjusted_camera_params[norm_idx][9:].reshape(3,1)
    R_mats = adjusted_R_mats
    t_vecs = adjusted_t_vecs

    for id, R_mat in R_mats.items():
        img_db[id + 1].qvec = rotmat2qvec(R_mat)

    for id, t_vec in t_vecs.items():
        img_db[id + 1].tvec = t_vec

    for pt3d_idx in range(len(pts3D_db)):
        pts3D_db[pt3d_idx].xyz = adjusted_points_3d[pt3d_idx]