import numpy as np
import scipy.optimize as optimize

from camera_treatment import reprojection

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

def reprojection_err(pts3d, pts2d1, pts2d2, P1, P2):
    pts2d1_proj, pts2d2_proj = reprojection(P1, P2, pts3d)
    
    err1_list = np.square(pts2d1 - pts2d1_proj)
    
    err1 = np.sum(np.square(pts2d1 - pts2d1_proj), axis=0)
    err2 = np.sum(np.square(pts2d2 - pts2d2_proj), axis=0)
    error = err1 + err2
    return error

def nonlinear_triangulation(K, RT1, RT2, pts3d, pts2d1, pts2d2): # Non linear triangulation reference https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_7_2-triangulation.pdf
    P1 = K @ RT1
    P2 = K @ RT2

    final_pts3d = []
    for i in range(pts3d.shape[1]):
        opt_params = optimize.least_squares(fun=reprojection_err, x0=pts3d[:, i], method='trf', args=[pts2d1[:, i], pts2d2[:, i], P1, P2])
        pt3d = opt_params.x
        final_pts3d.append(pt3d)
    
    return np.array(final_pts3d)