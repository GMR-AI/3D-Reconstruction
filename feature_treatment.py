import cv2
import os
import numpy as np
from output_classes import c_Image

def load_images_from_folder(folder):
    images = []
    img_db = {}
    id = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            img_db[id] = c_Image(iid=id, ifilename=filename, iqvec=-1, itvec=-1, icamera_id=1, ixys=np.ndarray(0), ipoint3D_ids=np.ndarray(0))
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

    matches = []
    for i in range(len(kp)):
        matches.append([])
        for j in range(len(kp)): # Only match each image with the rest, we don't need the full matrix
            matches_tmp = flann.knnMatch(des[i], des[j], k=2)

            # Lowe's ratio test
            good_matches = [m for m, n in matches_tmp if m.distance < 0.7 * n.distance]

            if len(good_matches) >= 4:
                # RANSAC to find homography and get inlier's mask
                pts1 = np.float32([kp[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp[j][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

                inliers = [good_matches[k] for k in range(len(good_matches)) if mask[k]==1]
                matches[i].append(inliers)
            else:
                matches[i].append([])
    
    return matches