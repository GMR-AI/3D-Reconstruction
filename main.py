from functions import *
from read_write_model import *







if __name__ == '__main__':
    cameras = {}
    punts3D = {}
    imagenes = {}

    ## COLMAP

    images, imagenes = load_images_from_folder('dinos')

    # Feature extraction
    kp, des = feature_extraction_set(images)

    # Feature matching
    matches = feature_matching_set(kp, des)

    # Inicializar los puntos 3D

    punts3D[t] = Point3D(
        id=t,
        xyz=p,
        rgb=images[0][index1[0], index1[1]],
        error=0,
        image_ids=[],
        point2D_idxs=1,
    )

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
    imagenes[1].qvec = rotmat2qvec(np.eye(3))
    imagenes[1].tvec = np.zeros((3,))

    K1, K2 = K, K
    RT2 = RT2s[0]

    pts3d = np.array([linear_triangulation(K, RT1, K, RT2, pts1, pts2) for RT2 in RT2s])

    RT2, pts_cloud = double_disambiguation(K, RT1, K, RT2s, pts1, pts2, pts3d)
    pts_cloud = pts_cloud[:, pts_cloud[2, :] > 0]

    for p in pts_cloud.T:
        t=len(punts3D)+1
        index1=kp[0][matches[(0,1)][t].queryIdx].pt
        index2=kp[1][matches[(0,1)][t].trainIdx].pt

    imagenes[2].qvec = rotmat2qvec(RT2[:3,:3])
    imagenes[2].tvec = RT2[:,-1]


    # Visualize 3D points
    import plotly.graph_objects as go

    # Assuming points_3D is your array of 3D points
    x = pts_cloud[0]
    y = pts_cloud[1]
    z = pts_cloud[2]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       marker=dict(size=2, color=z, colorscale='Viridis'))])

    fig.update_layout(scene=dict(xaxis_title='X',
                                 yaxis_title='Y',
                                 zaxis_title='Z'))

    fig.show()

    # Initialize general Projection matrix list, RTs list and 3d points list
    P_list = np.ndarray((images.shape[0], 3, 4))
    P_done = np.full((images.shape[0], 1), False)
    P_list[0], P_done[0] = K @ RT1, True
    P_list[1], P_done[1] = K @ RT2, True

    for pts in pts_cloud:
        plot_model(pts)

    # Escribir datos
    write_model(cameras, imagenes, punts3D, )