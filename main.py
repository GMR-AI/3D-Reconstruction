from functions import *
from output_classes import *

if __name__ == '__main__':
    cameras = {}
    punts3D = {}

    ## COLMAP

    images, img_db = load_images_from_folder('dinos')

    # Feature extraction
    kp, des = feature_extraction_set(images, img_db)

    # Feature matching
    matches = feature_matching_set(kp, des)

    # Inicializar camara
    height, width = images.shape[1:3]
    K = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])

    cameras[1] = Camera(
                    id=1,
                    model="OPENCV",
                    width=width,
                    height=height,
                    params=[2360, 2360, width / 2, height / 2, 0,0,0,0],
                )

    # Fundamental matrix
    pts1 = np.transpose([kp[0][m.queryIdx].pt for m in matches[(0, 1)]])
    pts2 = np.transpose([kp[1][m.trainIdx].pt for m in matches[(0, 1)]])

    F = eight_point_algorithm(pts1, pts2)

    # Essential matrix
    E = essential_from_fundamental(K, F, K)  # In this case, the same intrinsic values apply to all images

    # Get camera extrinsics from Essential matrix
    RT2s = pose_from_essential(E)

    # Define RT for camera 1 (center at world origin and matching orientation)
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    img_db[1].qvec = rotmat2qvec(np.eye(3))
    img_db[1].tvec = np.zeros((3,))

    K1, K2 = K, K
    RT2 = RT2s[0]

    pts3d = np.array([linear_triangulation(K, RT1, K, RT2, pts1, pts2) for RT2 in RT2s])

    RT2, pts_cloud = double_disambiguation(K, RT1, K, RT2s, pts1, pts2, pts3d)
    pts_cloud = pts_cloud[:, pts_cloud[2, :] > 0]

    for p in pts_cloud.T:
        point_idx = len(punts3D)
        index1 = kp[0][matches[(0,1)][point_idx].queryIdx].pt
        im_pts_idx = np.array([(m.trainIdx, j) for j in range(1, images.shape[0]) for m in matches[(0,j)] if m.queryIdx == matches[(0,1)][point_idx].queryIdx])
        punts3D[point_idx] = Point3D(
            id=point_idx,
            xyz=p,
            rgb=images[0, index1[1], index1[0], :],
            error=0,
            image_ids=im_pts_idx[:][1],
            point2D_idxs=im_pts_idx[:][0],
        )

    img_db[2].qvec = rotmat2qvec(RT2[:3,:3])
    img_db[2].tvec = RT2[:,-1]

    # Initialize general Projection matrix list, RTs list and 3d points list
    P_list = np.ndarray((images.shape[0], 3, 4))
    P_done = np.full((images.shape[0], 1), False)
    P_list[0], P_done[0] = K @ RT1, True
    P_list[1], P_done[1] = K @ RT2, True

    for pts in pts_cloud:
        plot_model(pts)

    # Escribir datos
    write_model(cameras, img_db, punts3D, )