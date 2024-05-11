from colmap_functions import *

import cv2

# Classes

class c_Point3D:
    def __init__(self, iid, ixyz, irgb, ierror=0, iimage_ids=[], ipoint2D_idxs=[]):
        self.id=iid
        self.xyz=ixyz
        self.rgb=irgb
        self.error=ierror
        self.image_ids=iimage_ids
        self.point2D_idxs=ipoint2D_idxs

    def to_tupla(self):
        return Point3D(
            id=self.id,
            xyz=self.xyz,
            rgb=self.rgb,
            error=self.error,
            image_ids=self.image_ids,
            point2D_idxs=self.point2D_idxs,
        )


class c_Image:
    def __init__(self, iid, ifilename, iqvec=None, itvec=None, icamera_id=1, ixys=None, ipoint3D_ids=None):
        self.id=iid
        self.filename=ifilename
        self.qvec=iqvec
        self.tvec=itvec
        self.camera_id=icamera_id
        self.xys=ixys
        self.point3D_idxs=ipoint3D_ids

    def to_tupla(self):
        return Image(
            id=self.id,
            name=self.filename,
            qvec=self.qvec,
            tvec=self.tvec,
            camera_id=self.camera_id,
            xys=self.xys,
            point3D_ids=self.point3D_idxs)


def fill_pts3d(img_db: dict[int, c_Image], pts3D_db: dict[int, c_Point3D], pts3d: np.ndarray, matches: list[list], images: np.ndarray, idx1: int, idx2: int):
    """
    Add all the 3D points to the points database and relate them with the image database.

    img_db: image colmap database.
    pts3D_db: 3D points colmap database.
    pts3d: 3xN 3D points to add.
    kp_img: tuple of keypoints from image 1.
    matches: list matrix of matches between all images.
    images: NxHeightxWidth array of images.
    idx1: index of image 1.
    idx2: index of image 2.
    """
    num_pt = 0
    for p in pts3d.T:
        if num_pt >= len(matches[idx1][idx2]):
            return
        kp_idx1 = matches[idx1][idx2][num_pt].queryIdx
        kp_idx2 = matches[idx1][idx2][num_pt].trainIdx

        if img_db[idx1+1].point3D_idxs[kp_idx1] != -1:
            num_pt += 1
            if num_pt >= len(matches[idx1][idx2]):
                return
            kp_idx1 = matches[idx1][idx2][num_pt].queryIdx
            kp_idx2 = matches[idx1][idx2][num_pt].trainIdx

        pt_id = len(pts3D_db)
        pixel_coords = np.array(img_db[idx1+1].xys[kp_idx1]).astype(int)
        im_pts_id = []
        for i in range(images.shape[0]):
            if i == idx1: continue

            if i < idx1:
                for m in matches[i][idx1]:
                    if m.trainIdx == kp_idx1:
                        im_pts_id.append([i+1, m.queryIdx])
                        img_db[i+1].point3D_idxs[m.queryIdx] = pt_id

            if i > idx1:
                for m in matches[idx1][i]:
                    if m.queryIdx == kp_idx1:
                        im_pts_id.append([i+1, m.trainIdx])
                        img_db[i+1].point3D_idxs[m.trainIdx] = pt_id
            
        for i in range(images.shape[0]):
            if i == idx2: continue

            if i < idx2:
                for m in matches[i][idx2]:
                    if m.trainIdx == kp_idx2:
                        im_pts_id.append([i+1, m.queryIdx])
                        img_db[i+1].point3D_idxs[m.queryIdx] = pt_id
            
            if i > idx2:
                for m in matches[idx2][i]:
                    if m.queryIdx == kp_idx2:
                        im_pts_id.append([i+1, m.trainIdx])
                        img_db[i+1].point3D_idxs[m.trainIdx] = pt_id
        
        im_pts_id = np.array(im_pts_id)

        pts3D_db[pt_id] = c_Point3D(
            iid=pt_id,
            ixyz=p,
            irgb=images[idx1][pixel_coords[1]][pixel_coords[0]],
            ierror=0,
            iimage_ids=im_pts_id[:, 0],
            ipoint2D_idxs=im_pts_id[:, 1]
        )
        for im_pts in im_pts_id:
            img_db[im_pts[0]].point3D_idxs[im_pts[1]] = pt_id
        num_pt += 1