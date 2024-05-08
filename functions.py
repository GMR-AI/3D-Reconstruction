import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from output_classes import *

import cv2
import os


# ## Load Dataset and Setup Pipeline

def load_images_from_folder(folder):
    images = []
    img_db = {}
    id = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            img_db[id] = c_Image(iid=id, ifilename=filename, iqvec=np.zeros((4,)), itvec=np.zeros((3,)), icamera_id=1, ixys=np.ndarray(0), ipoint3D_ids=np.ndarray(0))
            id+=1
    return np.array(images), img_db

def feature_extraction_set(images, im_db):
    sift = cv2.SIFT_create()

    kp, des = [], []
    id = 1
    for im in images:
        kp_tmp, des_tmp = sift.detectAndCompute(im, None) # This assumes the extraction method to be from the CV2 library
        im_db[id].xys = np.array([k.pt for k in kp_tmp])
        im_db[id].point3D_idxs = np.full((len(kp_tmp),), -1)
        kp.append(kp_tmp)
        des.append(des_tmp)
        id += 1
    return kp, des # Can't turn them into a np array since their shape can be inhomogeneous

def feature_matching_set(kp, des):
    # Initialize FLANN matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = {} # Dict for easier access to each match
    for i in range(len(kp)):
        for j in range(i+1, len(kp)): # Only match each image with the rest, we don't need the full matrix
            matches_tmp = flann.knnMatch(des[i], des[j], k=2)

            # Lowe's ratio test
            good_matches = [m for m, n in matches_tmp if m.distance < 0.7 * n.distance]

            if len(good_matches) >= 4:
                # RANSAC to find homography and get inlier's mask
                pts1 = np.float32([kp[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp[j][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

                inliers = [good_matches[k] for k in range(len(good_matches)) if mask[k]==1]
                matches[(i,j)] = inliers
            else:
                matches[(i,j)] = []
    
    return matches


def normalize(pts):
    x_mean = np.mean(pts[:, 0])
    y_mean = np.mean(pts[:, 1])
    sigma = np.mean(np.sqrt((pts[:, 0] - x_mean) ** 2 + (pts[:, 1] - y_mean) ** 2))
    M = np.sqrt(2) / sigma
    T = np.array([
        [M, 0, -M * x_mean],
        [0, M, -M * y_mean],
        [0, 0, 1]
    ])
    return T

def eight_point_algorithm(pts1, pts2):

    pts1_homo = np.vstack((pts1, np.ones(pts1.shape[1]))).T
    pts2_homo = np.vstack((pts2, np.ones(pts2.shape[1]))).T

    # Normalization
    T = normalize(pts1_homo)
    T_prime = normalize(pts2_homo)


    pts1_homo = (T @ pts1_homo.T).T
    pts2_homo = (T_prime @ pts2_homo.T).T

    # x2.T*F*x1=0
    # A*f=0, f is F flattened into a 1D array
    

    # Create A
    A = np.zeros((pts1.shape[1], 9))
    for i in range(pts1.shape[1]):
        A[i] = np.array([
            pts1_homo[i,0]*pts2_homo[i,0], pts1_homo[i,1]*pts2_homo[i,0], pts1_homo[i,2]*pts2_homo[i,0],
            pts1_homo[i,0]*pts2_homo[i,1], pts1_homo[i,1]*pts2_homo[i,1], pts1_homo[i,2]*pts2_homo[i,1],
            pts1_homo[i,0]*pts2_homo[i,2], pts1_homo[i,1]*pts2_homo[i,2], pts1_homo[i,2]*pts2_homo[i,2]
            ])
    
    # Solve Af=0 using svd
    U,S,Vt = np.linalg.svd(A)
    F = Vt[-1,:].reshape((3,3))

    # Enforce rank2 constraint
    U,S,Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    F = T_prime.T @ F @ T
    return F


def essential_from_fundamental(K1, F, K2):
    return K1.T @ F @ K2


def pose_from_essential(E):
    U,_,Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    Rs = [U @ W @ Vt, U @ W.T @ Vt]
    for i in range(len(Rs)):
        if np.linalg.det(Rs[i]) < 0:
            Rs[i] = Rs[i] * -1

    # Array with all possible camera poses (extrinsics)
    RTs = np.array([
        np.hstack((Rs[0], U[:, 2, np.newaxis])),
        np.hstack((Rs[0], -U[:, 2, np.newaxis])),
        np.hstack((Rs[1], U[:, 2, np.newaxis])),
        np.hstack((Rs[1], -U[:, 2, np.newaxis])),
    ])

    return RTs


def linear_triangulation(K1, RT1, K2, RT2, pts1, pts2):
    # First, set all points to homogeneous
    pts1_homo = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2_homo = np.vstack((pts2, np.ones(pts2.shape[1])))

    # Calculate every projection matrix
    P1 = K1 @ RT1
    P2 = K2 @ RT2

    # Solve using svd
    pts3d = np.zeros((3, pts1.shape[1]))
    for i in range(pts1.shape[1]):
        A = np.array([pts1_homo[1,i]*P1[2,:] - P1[1,:],
            P1[0,:] - pts1_homo[0,i]*P1[2,:],
            pts2_homo[1,i]*P2[2,:] - P2[1,:],
            P2[0,:] - pts2_homo[0,i]*P2[2,:]])
        ATA = A.T @ A
        _, _, Vt = np.linalg.svd(ATA)
        pts3d[:, i] = Vt[-1, :3]/Vt[-1, -1]
    
    return pts3d

def linear_triangulation2(P1, P2, pts1, pts2):
    # First, set all points to homogeneous
    pts1_homo = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2_homo = np.vstack((pts2, np.ones(pts2.shape[1])))

    # Solve using svd
    pts3d = np.zeros((3, pts1.shape[1]))
    for i in range(pts1.shape[1]):
        A = np.array([pts1_homo[1,i]*P1[2,:] - P1[1,:],
                      P1[0,:] - pts1_homo[0,i]*P1[2,:],
                      pts2_homo[1,i]*P2[2,:] - P2[1,:],
                      P2[0,:] - pts2_homo[0,i]*P2[2,:]])
        ATA = A.T @ A
        _, _, Vt = np.linalg.svd(ATA)
        pts3d[:, i] = Vt[-1, :3]/Vt[-1, -1]
    
    return pts3d



def reprojection(P1, P2, pts3d):
    pts3d_homo = np.vstack((pts3d, np.ones(pts3d.shape[1])))
    pts2d1_homo = np.dot(P1, pts3d_homo)
    pts2d2_homo = np.dot(P2, pts3d_homo)
    return pts2d1_homo/pts2d1_homo[-1], pts2d2_homo/pts2d2_homo[-1]

def double_disambiguation(K1, RT1, K2, RT2s, pts1, pts2, pts3d):
    max_positive_z = 0
    min_error = np.finfo('float').max
    best_RT = None
    best_pts3d = None
    P1 = K1 @ RT1

    pts1_homo = np.vstack((pts1, np.ones(pts1.shape[1])))
    pts2_homo = np.vstack((pts2, np.ones(pts2.shape[1])))

    for i in range(RT2s.shape[0]):
        P2 = K2 @ RT2s[i]
        num_positive_z = np.sum(pts3d[i][2, :] > 0)
        re1_pts2, re2_pts2 = reprojection(P1, P2, pts3d[i])

        err1 = np.sum(np.square(re1_pts2 - pts1_homo))
        err2 = np.sum(np.square(re2_pts2 - pts2_homo))

        err = err1 + err2

        if num_positive_z >= max_positive_z and err < min_error:
            max_positive_z = num_positive_z
            min_error = err
            best_RT = RT2s[i]
            best_pts3d = pts3d[i]
    
    return best_RT, best_pts3d

def calculate_projection_matrix(K, pts3d, pts2d):
    _, rod, T, _ = cv2.solvePnPRansac(pts3d.T, pts2d.T, K, None)#, flags=cv2.SOLVEPNP_P3P)
    R = cv2.Rodrigues(rod)[0]
    if np.linalg.det(R) < 0:
        R = R * -1
    P = K @ np.hstack((R, T))
    return P


def plot_model(pts_cloud):
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