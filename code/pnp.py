import random
import cv2
import numpy as np

def following_img_reconstruction(n_imgs: int, init_pair: tuple[int], processed_imgs: list[int], unprocessed_imgs: list[int]):
    """
    Get the next image to process and the image it will triangulate with.

    n_imgs: total number of images images.
    init_pair: initial image pair of the reconstruction process.
    processed_imgs: list with the indexes of the already processed images.
    unprocessed_imgs: list with the indexes of the unprocessed images.
    """

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


def calculate_projection_matrix(K: np.ndarray, pts2d: np.ndarray, pts3d: np.ndarray):
    """
    Calculate the camera pose from solving PnP using cv2.

    K: 3x3 matrix with camera intrinsics.
    pts2d: 2xN 2D points corresponding to the projection of the 3D points.
    pts3d: 3xN 3D points corresponding to the 2D points.
    """
    _, rod, T, _ = cv2.solvePnPRansac(pts3d.T, pts2d.T, K, None)#, flags=cv2.SOLVEPNP_P3P)
    R = cv2.Rodrigues(rod)[0]
    if np.linalg.det(R) < 0:
        R = R * -1
    return R, T