import os
import cv2
import numpy as np

import colmap_functions as cf
import output_classes as oc

def load_images_from_folder(path: str, img_db: dict):
    """
    Load all the images from the specified folder.

    path: relative path to the folder
    img_db: image colmap database to initialize
    """

    images = []
    id = 1
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
            img_db[id] = oc.c_Image(iid=id, ifilename=filename)
            id += 1
    return np.array(images)


def feature_extraction(images: np.ndarray, img_db: dict[int, oc.c_Image]):
    """
    Feature extraction for all images using SIFT.

    images: NxHeightxWidth ndarray of images to extract features from.
    img_db: image colmap database.
    """

    sift = cv2.SIFT_create()
    kp, des = [], []
    id = 1
    for im in images:
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        kp_tmp, des_tmp = sift.detectAndCompute(gray_im, None)
        img_db[id].xys = np.array([k.pt for k in kp_tmp])
        img_db[id].point3D_idxs = np.full(len(kp_tmp), -1)
        kp.append(kp_tmp)
        des.append(des_tmp)
        id += 1
    return kp, des


def feature_matching(kp: list[tuple[cv2.KeyPoint]], des: list[cv2.DescriptorMatcher]):
    """
    Feature matching for all images using FLANN.

    kp: list of tuples of keypoints of all images.
    des: list of descriptors of every image.
    """
    matcher = cv2.BFMatcher(cv2.NORM_L1)
    matches = []
    n_imgs = len(kp)

    for i in range(n_imgs):
        matches.append([])
        for j in range(n_imgs):
            if j <= i: matches[i].append(None)

            else:
                match=[]
                m = matcher.knnMatch(des[i], des[j], k=2)
                for k in range(len(m)):
                    try:
                        if m[k][0].distance < 0.7*m[k][1].distance:
                            match.append(m[k][0])
                    except:
                        continue
                matches[i].append(match)
    
    # Remove outliers
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue
            kpts_i = []
            kpts_j = []
            for k in range(len(matches[i][j])):
                kpts_i.append(kp[i][matches[i][j][k].queryIdx].pt)
                kpts_j.append(kp[j][matches[i][j][k].trainIdx].pt)
            kpts_i = np.int32(kpts_i)
            kpts_j = np.int32(kpts_j)
            F, mask = cv2.findFundamentalMat(kpts_i, kpts_j, cv2.FM_RANSAC, ransacReprojThreshold=3)
            if np.linalg.det(F) > 1e-7: raise ValueError(f"Bad F_mat between images: {i}, {j}. Determinant: {np.linalg.det(F)}")
            matches[i][j] = np.array(matches[i][j])
            if mask is None:
                matches[i][j] = []
                continue
            matches[i][j] = matches[i][j][mask.ravel() == 1]
            matches[i][j] = list(matches[i][j])

            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue

    return matches


def create_adj_matrix(matches: list[list]):
    """
    Create an adjacency matrix connecting all images that have at least one match.

    matches: list matrix of matches between all images.
    """
    matches_shape = (len(matches), len(matches))
    adj_matrix = np.zeros(matches_shape)
    for i in range(matches_shape[0]):
        for j in range(matches_shape[1]):
            if i < j and len(matches[i][j]) > 0:
                adj_matrix[i][j] = 1
    return adj_matrix


def matching_keypoints_old(idx1: int, idx2: int, kp: list[tuple[cv2.KeyPoint]], matches: list[list]):
    """
    Get only the keypoints with matches between the specified images.

    idx1: index of image 1.
    idx2: index of image 2.
    kp: list of tuples of keypoints of all images.
    matches: list matrix of matches between all images.

    returns:
        kp1: 2xN 2D keypoints for image 1.
        kp2: 2xN 2D keypoints for image 2.
    """

    if matches[idx1][idx2] is None: raise ValueError('None matches between the specified images. idx1 must be lower than idx2 (idx1<idx2)')
    kp1, kp2 = [], []

    for m in matches[idx1][idx2]:
        kp1.append(kp[idx1][m.queryIdx].pt)
        kp2.append(kp[idx2][m.trainIdx].pt)
    
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)

    if len(kp1.shape) < 2:
        kp1 = np.expand_dims(kp1, axis=1)
        kp2 = np.expand_dims(kp2, axis=1)
    
    return kp1.T, kp2.T


def matching_keypoints(img_db: dict[int, oc.c_Image], matches: list[list], id1: int, id2: int):
    """
    Get the matching keypoints between the specified images if they have not been triangulated yet.

    img_db: image colmap database.
    matches: list matrix of matches between all images.
    id1: id of the first image (remember, the image database goes from 1 to N)
    if2: id of the second image (remember, the image database goes from 1 to N)
    """

    pts2d1, pts2d2 = [], []
    for m in matches[id1-1][id2-1]:
        if img_db[id1].point3D_idxs[m.queryIdx] == img_db[id2].point3D_idxs[m.trainIdx] == -1:
            pts2d1.append(img_db[id1].xys[m.queryIdx])
            pts2d2.append(img_db[id2].xys[m.trainIdx])
    pts2d1 = np.transpose(pts2d1)
    pts2d2 = np.transpose(pts2d2)

    return pts2d1, pts2d2



if __name__ == '__main__':
    img_db = {}
    images = load_images_from_folder('dinos', img_db)
    kp, des = feature_extraction(images, img_db)
    print(type(kp[0][0]))