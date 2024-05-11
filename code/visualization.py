import plotly.graph_objects as go

import numpy as np

def plot_model(points3d_with_views):
    # Extract the 3D points from points3d_with_views
    pts_cloud = np.array([pt3.xyz for pt3 in points3d_with_views.values()])
    
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