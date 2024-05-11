import numpy as np

import features as ft
import two_sfm as ts
import colmap_functions as cf
import output_classes as oc
import triangulation as tr
import bundle_adjustment as ba
import pnp

from typing import Literal
from visualization import plot_model

def main(path: str = 'dinos', method: Literal['cv2', 'custom'] = 'cv2'):
    # Initialize colmap output dictionaries
    cameras_db: dict[int, cf.Camera] = {}
    img_db: dict[int, oc.c_Image] = {}
    pts3D_db: dict[int, oc.c_Point3D] = {}

    # Load, extract and match all images
    images = ft.load_images_from_folder(path, img_db)
    kp, des = ft.feature_extraction(images, img_db)
    matches = ft.feature_matching(kp, des)
    adj_matrix = ft.create_adj_matrix(matches)

    # Initialize camera
    height, width = images.shape[1:3]
    K = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])
    cameras_db[1] = cf.Camera(
                id=1,
                model="OPENCV",
                width=width,
                height=height,
                params=[2360, 2360, width / 2, height / 2, 0,0,0,0],
            )
    
    # Calculate initial camera poses
    starting_pair = ts.starting_img_set(adj_matrix, img_db, matches, K)

    R1, t1, R2, t2, pts3d = ts.get_camera_pose(img_db, matches, K, starting_pair[0], starting_pair[1], method=method)
    img_db[starting_pair[0] + 1].qvec = cf.rotmat2qvec(R1)
    img_db[starting_pair[0] + 1].tvec = t1
    img_db[starting_pair[1] + 1].qvec = cf.rotmat2qvec(R2)
    img_db[starting_pair[1] + 1].tvec = t2

    oc.fill_pts3d(img_db, pts3D_db, pts3d, matches, images, starting_pair[0], starting_pair[1])

    plot_model(pts3D_db) # Check the model so far

    processed_imgs = [starting_pair[0], starting_pair[1]]
    unprocessed_imgs = [i for i in range(len(images)) if not i in processed_imgs]

    # Main PnP loop
    BA_chkpts = [3,4,5,6] + [int(6*(1.34**i)) for i in range(25)]
    while len(unprocessed_imgs) > 0:
        processed_idx, unprocessed_idx, prepend = pnp.following_img_reconstruction(images.shape[0], starting_pair, processed_imgs, unprocessed_imgs)
        pts3d_idx = img_db[unprocessed_idx + 1].point3D_idxs != -1
        if np.sum(pts3d_idx) < 12:
            continue
        print('processing')

        pts3d = np.array([pts3D_db[idx].xyz for idx in img_db[unprocessed_idx + 1].point3D_idxs[pts3d_idx]]).T
        pts2d = img_db[unprocessed_idx + 1].xys[pts3d_idx].T
        
        R1 = cf.qvec2rotmat(img_db[processed_idx + 1].qvec)
        t1 = img_db[processed_idx + 1].tvec
        R2, t2 = pnp.calculate_projection_matrix(K, pts2d, pts3d)
        
        img_db[unprocessed_idx + 1].qvec = cf.rotmat2qvec(R2)
        img_db[unprocessed_idx + 1].tvec = t2

        if prepend: processed_imgs.insert(0, unprocessed_idx)
        else: processed_imgs.append(unprocessed_idx)
        unprocessed_imgs.remove(unprocessed_idx)
        _, _, perc_inliers = tr.reprojection_error(K, R2, t2, pts2d, pts3d)
        
        if processed_idx < unprocessed_idx:
            kpts1, kpts2 = ft.matching_keypoints(img_db, matches, processed_idx + 1, unprocessed_idx + 1)
            if kpts1.shape[1] > 0:
                pts3d, avg_tri_err_l, avg_tri_err_r = tr.triangulate_and_reproject(K, R1, t1, R2, t2, kpts1, kpts2, method=method)
                oc.fill_pts3d(img_db, pts3D_db, pts3d, matches, images, processed_idx, unprocessed_idx)
        else:
            kpts2, kpts1 = ft.matching_keypoints(img_db, matches, unprocessed_idx + 1, processed_idx + 1)
            if kpts2.shape[1] > 0:
                pts3d, avg_tri_err_l, avg_tri_err_r = tr.triangulate_and_reproject(K, R2, t2, R1, t1, kpts2, kpts1, method=method)
                oc.fill_pts3d(img_db, pts3D_db, pts3d, matches, images, unprocessed_idx, processed_idx)
        
        
        # Bundle adjustment
        if 0.8 < perc_inliers < 0.95 or 5 < avg_tri_err_l < 10 or 5 < avg_tri_err_r < 10: 
            ba.bundleAd(pts3D_db, img_db, processed_imgs, K, ftol=1e0)
            
        if len(processed_imgs) in BA_chkpts or len(unprocessed_imgs) == 0 or perc_inliers <= 0.8 or avg_tri_err_l >= 10 or avg_tri_err_r >= 10:
            ba.bundleAd(pts3D_db, img_db, processed_imgs, K, ftol=1e-1)
    
    plot_model(pts3D_db)

    pts3D_output = {}
    for key, pts3D_colmap in pts3D_db.items():
        pts3D_output[key] = pts3D_colmap.to_tupla()
    
    img_output = {}
    for key, img_colmap in img_db.items():
        img_output[key] = img_colmap.to_tupla()
    
    cf.write_cameras_binary(cameras_db, 'cameras_binary')
    cf.write_images_binary(img_output, 'images_binary')
    cf.write_points3D_binary(pts3D_output, 'points3D_binary')


if __name__ == '__main__':
    main(path='dinos', method='cv2')