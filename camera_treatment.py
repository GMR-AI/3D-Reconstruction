import numpy as np

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

def reprojection(P1, P2, pts3d):
    try:
        num_pts3d = pts3d.shape[1]
    except:
        num_pts3d = 1
    
    local_pts3d = pts3d
    if num_pts3d == 1:
        local_pts3d = pts3d[:,np.newaxis]
    
    pts3d_homo = np.vstack((local_pts3d, np.ones(num_pts3d)))
    pts2d1_homo = P1 @ pts3d_homo
    pts2d2_homo = P2 @ pts3d_homo
    return np.squeeze(pts2d1_homo[:2]/pts2d1_homo[-1]), np.squeeze(pts2d2_homo[:2]/pts2d2_homo[-1])

def double_disambiguation(K1, RT1, K2, RT2s, pts1, pts2, pts3d):
    max_positive_z = 0
    min_error = np.finfo('float').max
    best_RT = None
    best_pts3d = None
    P1 = K1 @ RT1

    for i in range(RT2s.shape[0]):
        P2 = K2 @ RT2s[i]
        num_positive_z = np.sum(pts3d[i][2, :] > 0)
        re1_pts2, re2_pts2 = reprojection(P1, P2, pts3d[i])

        err1 = np.sum(np.square(re1_pts2 - pts1))
        err2 = np.sum(np.square(re2_pts2 - pts2))

        err = err1 + err2

        if num_positive_z >= max_positive_z and err < min_error:
            max_positive_z = num_positive_z
            min_error = err
            best_RT = RT2s[i]
            best_pts3d = pts3d[i]
    
    return best_RT, best_pts3d

def calculate_projection_matrix(K, pts3d, pts2d):
    _, rod, T, _ = cv2.solvePnPRansac(pts3d, pts2d, K, None)#, flags=cv2.SOLVEPNP_P3P)
    R = cv2.Rodrigues(rod)[0]
    if np.linalg.det(R) < 0:
        R = R * -1
    P = K @ np.hstack((R, T))
    return P