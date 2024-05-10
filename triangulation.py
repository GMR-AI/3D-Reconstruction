import numpy as np
import scipy.optimize as optimize
import cv2
import random
from camera_treatment import reprojection_error


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
        if Vt[-1, -1] != 0:
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
        if Vt[-1, -1] != 0:
            pts3d[:, i] = Vt[-1, :3]/Vt[-1, -1]
    
    return pts3d

def following_img_reconstruction(n_imgs, init_pair, processed_imgs, unprocessed_imgs):

    if len(unprocessed_imgs) == 0: raise ValueError('Should not check next image to resect if all have been resected already!')
    outer_interval = False
    if init_pair[1] - init_pair[0] > n_imgs/2 : outer_interval = True #initial pair straddles "end" of the circle (ie if init pair is idxs (0, 49) for 50 images)

    init_arc = init_pair[1] - init_pair[0] + 1 # Number of images between and including initial pair

    #fill in images between initial pair
    if len(processed_imgs) < init_arc:
        if outer_interval == False: idx = processed_imgs[-2] + 1
        else: idx = processed_imgs[-1] + 1
        while True:
            if idx not in processed_imgs:
                prepend = True
                unprocessed_idx = idx
                processed_idx = random.choice(processed_imgs)
                return processed_idx, unprocessed_idx, prepend
            idx = idx + 1 % n_imgs

    extensions = len(processed_imgs) - init_arc # How many images have been resected after the initial arc
    if outer_interval == True: #smaller init_idx should be increased and larger decreased
        if extensions % 2 == 0:
            unprocessed_idx = (init_pair[0] + int(extensions/2) + 1) % n_imgs
            processed_idx = (unprocessed_idx - 1) % n_imgs
        else:
            unprocessed_idx = (init_pair[1] - int(extensions/2) - 1) % n_imgs
            processed_idx = (unprocessed_idx + 1) % n_imgs
    else:
        if extensions % 2 == 0:
            unprocessed_idx = (init_pair[1] + int(extensions/2) + 1) % n_imgs
            processed_idx = (unprocessed_idx - 1) % n_imgs
        else:
            unprocessed_idx = (init_pair[0] - int(extensions/2) - 1) % n_imgs
            processed_idx = (unprocessed_idx + 1) % n_imgs

    prepend = False
    return processed_idx, unprocessed_idx, prepend


def alignment_of_kpts(img_db, matches, i, j):
    idxs_i = [m.queryIdx for m in matches[i][j]]
    kpts_i_idxs = img_db[i].point3D_idxs[idxs_i] != -1
    kpts_i = img_db[i].xys[kpts_i_idxs]

    idxs_j = [m.queryIdx for m in matches[i][j]]
    kpts_j_idxs = img_db[j].point3D_idxs[idxs_j] != -1
    kpts_j = img_db[j].xys[kpts_j_idxs]

    return kpts_i, kpts_j, kpts_i_idxs, kpts_j_idxs


def triangulate_and_reproject(K1, RT1, K2, RT2, pts2d1, pts2d2, reproject = True):
    pts3d = linear_triangulation(K1, RT1, K2, RT2, pts2d1, pts2d2)

    if reproject:
        _, avg_error1, _ = reprojection_error(K1, RT1, pts3d, pts2d1)
        _, avg_error2, _ = reprojection_error(K2, RT2, pts3d, pts2d2)

        return pts3d, avg_error1, avg_error2

    return pts3d
