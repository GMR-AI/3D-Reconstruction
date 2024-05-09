import numpy as np

from feature_treatment import feature_extraction_set, feature_matching_set
from camera_treatment import eight_point_algorithm, essential_from_fundamental, pose_from_essential, double_disambiguation
from triangulation import linear_triangulation
from utils import plot_model

def two_images_sfm(im1, im2, K):
    # Feature extraction
    kp, des = feature_extraction_set([im1, im2])

    # Feature matching
    matches = feature_matching_set(kp, des)

    # Fundamental matrix
    pts1 = np.transpose([kp[0][m.queryIdx].pt for m in matches[(0,1)]])
    pts2 = np.transpose([kp[1][m.trainIdx].pt for m in matches[(0,1)]])

    F = eight_point_algorithm(pts1, pts2)

    # Essential matrix
    E = essential_from_fundamental(K, F, K) # In this case, the same intrinsic values apply to all images

    # Get camera extrinsics from Essential matrix
    RT2s = pose_from_essential(E)

    # Define RT for camera 1 (center at world origin and matching orientation)
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    
    RT2 = RT2s[0]

    pts3d = np.array([linear_triangulation(K, RT1, K, RT2, pts1, pts2) for RT2 in RT2s])

    RT2, pts_cloud = double_disambiguation(K, RT1, K, RT2s, pts1, pts2, pts3d)

    plot_model(pts_cloud)

    # Initialize general Projection matrix list, RTs list and 3d points list

    return K @ RT1, K @ RT2, pts_cloud