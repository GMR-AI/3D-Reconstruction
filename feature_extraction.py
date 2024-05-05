import cv2
from ransacImplement import ransac_matrix
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # add file types as needed
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img_rgb = img
                images.append(img_rgb)
    return images


def multi_feature_extraction(images):
    kp = []
    des = []
    for image in images:
        kptmp, destmp = feature_extraction(image)
        kp.append(kptmp)
        des.append(destmp)
    return kp, des


def multi_feature_matching(des, window_size):
    simple_matches = []
    multi_matches = []
    for i in range(len(des)):
        multi_matches.append([])
        for j in range(i-int(window_size/2), i+int(window_size/2)):
            j = j % len(des)
            if i == j:
                continue
            matchtmp = feature_matching(des[i], des[j])
            if j == (i + 1) % len(des):
                simple_matches.append(matchtmp)
            multi_matches[i].append(matchtmp)
    return simple_matches, multi_matches


def feature_extraction(image):
    grayimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints_orb, descriptors_orb = orb.detectAndCompute(grayimage, None)
    return keypoints_orb, descriptors_orb


def feature_matching(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


if __name__ == '__main__':
    folder = 'dataset'
    images = load_images_from_folder(folder)
    kp, des = multi_feature_extraction(images)
    matches = multi_feature_matching(des, window_size=20)

    fundamental_matrices = ransac_matrix(matches, kp)
    print(fundamental_matrices)