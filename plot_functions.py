import plotly.express as px


def plot_frame(frame_data):
    """
    Plot frame data as an image.
    """
    return px.imshow(frame_data)


def plot_roi(roi_data):
    """
    Plot ROI (Region of Interest) data as an image.
    """
    return px.imshow(roi_data)


def plot_rgb_channels(rgb_data):
    """
    Plot RGB channels.
    """
    return {
        "data": [
            {"x": list(range(256)), "y": rgb_data[0], "type": "scatter", "name": "R"},
            {"x": list(range(256)), "y": rgb_data[1], "type": "scatter", "name": "G"},
            {"x": list(range(256)), "y": rgb_data[2], "type": "scatter", "name": "B"},
        ],
        "layout": {"title": "RGB Channels"},
    }


def plot_head_pose(head_pose):
    """
    Plot Head Pose Angles.
    """
    return {
        "data": [
            {"x": list(range(256)), "y": head_pose[0], "type": "scatter", "name": "x"},
            {"x": list(range(256)), "y": head_pose[1], "type": "scatter", "name": "y"},
            {"x": list(range(256)), "y": head_pose[2], "type": "scatter", "name": "z"},
        ],
        "layout": {"title": "3D Head Pose angles"},
    }
