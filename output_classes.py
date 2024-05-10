from read_write_model import *

# Classes

class c_Point3D:
    def __init__(self, iid, ixyz, irgb, ierror=0, iimage_ids=[], ipoint2D_idxs=[]):
        self.id=iid,
        self.xyz=ixyz,
        self.rgb=irgb,
        self.error=ierror,
        self.image_ids=iimage_ids,
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
    def __init__(self, iid, ifilename, iqvec, itvec, icamera_id=1, ixys=np.ndarray(0), ipoint3D_ids=np.ndarray(0)):
        self.id=iid,
        self.filename=ifilename,
        self.qvec=iqvec,
        self.tvec=itvec,
        self.camera_id=icamera_id,
        self.xys=ixys,
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