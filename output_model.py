# Preparar COLMAP output para nerf studio:
# https://docs.nerf.studio/quickstart/data_conventions.html
import json
import os

# Usar el metodo de sparse reconstruction del modelo oficial de COLMAP
# https://colmap.github.io/format.html
from read_write_model import *
def write_nerfstudio_data(cameras, image_paths, keypoints, points3D, output_dir):
    """
    Write the parsed data from COLMAP to a format suitable for NERFStudio.
    """
    # Write camera intrinsics
    intrinsics = {
        "camera_model": "OPENCV",  # Assuming all images share the same intrinsics
        "fl_x": cameras[0]['params'][0],
        "fl_y": cameras[0]['params'][1],
        "cx": cameras[0]['params'][2],
        "cy": cameras[0]['params'][3],
        "w": cameras[0]['width'],
        "h": cameras[0]['height'],
        "k1": cameras[0]['params'][4],
        "k2": cameras[0]['params'][5],
        "p1": cameras[0]['params'][6],
        "p2": cameras[0]['params'][7]
    }

    # Write per-frame extrinsics
    frames = []
    for idx, image_path in enumerate(image_paths):
        frame_data = {
            "file_path": os.path.basename(image_path),
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
        frames.append(frame_data)

    # Write to JSON file
    output_data = {
        "camera_intrinsics": intrinsics,
        "frames": frames
    }

    output_file = os.path.join(output_dir, "nerfstudio_data.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)