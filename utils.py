import numpy as np
import plotly.graph_objects as go

from output_classes import Point3D


def plot_model(points3d_with_views):
    # Extract the 3D points from points3d_with_views
    pts_cloud = np.array([pt3.xyz for pt3 in points3d_with_views.values() if np.abs(np.sum(pt3.xyz)) < 200])

    # Assuming points_3D is your array of 3D points
    x = pts_cloud[:, 0]
    y = pts_cloud[:, 1]
    z = pts_cloud[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def fill_pts3d(img_db: dict, punts3D_db: dict, pts_cloud: np.ndarray, kp_img: np.ndarray, matches: list[list], images: np.ndarray, idx1: int, idx2: int):
    """
    testo
    """
    num_pt = 0
    for p in pts_cloud.T:
        if img_db[idx1+1].point3D_idxs[matches[idx1][idx2][num_pt].queryIdx] != -1:
            num_pt += 1
            continue
        point_id = len(punts3D_db)
        im_pos = kp_img[matches[idx1][idx2][num_pt].queryIdx].pt
        im_pts_idx = [[idx1+1, matches[idx1][idx2][num_pt].queryIdx]]
        for i in range(images.shape[0]):
            if i == idx1:
                continue

            for m in matches[idx1][i]:
                if m.queryIdx == matches[idx1][idx2][num_pt].queryIdx:
                    im_pts_idx.append([i+1, m.trainIdx])
        im_pts_idx = np.array(im_pts_idx)
        #im_pts_idx = np.concatenate(([(idx1+1, matches[idx1][idx2][num_pt].queryIdx)], [(j+1, m.trainIdx) for j in range(idx1+1, images.shape[0]) for m in matches[idx1][j] if m.queryIdx == matches[idx1][idx2][num_pt].queryIdx]))
        punts3D_db[point_id] = Point3D(
            id=point_id,
            xyz=p,
            rgb=images[0, round(im_pos[1]), round(im_pos[0]), :],
            error=0,        
            image_ids=im_pts_idx[:, 0],
            point2D_idxs=im_pts_idx[:, 1],
        )
        for im_pts in im_pts_idx:
            img_db[im_pts[0]].point3D_idxs[im_pts[1]] = point_id
        num_pt += 1


def connect_images(matches):
    adj_matrix = [[1 if i < j and len(matches[i][j]) >= 4 else 0 for j in range(len(matches[i]))] for i in range(len(matches))]
    return np.array(adj_matrix)