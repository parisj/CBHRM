import plotly.express as px


def plot_frame(frame_data):
    """
    Plot frame data as an image.
    """
    return px.imshow(frame_data, binary_string=True)


def plot_roi(roi_data):
    """
    Plot ROI (Region of Interest) data as an image.
    """
    return px.imshow(roi_data, binary_string=True)


def plot_rgb_channels(rgb_data):
    """
    Plot RGB channels.
    """
    return {
        "data": [
            {
                "x": list(range(256)),
                "y": list(rgb_data[0]),
                "type": "scatter",
                "name": "R",
                "marker": {"color": "red"},
            },
            {
                "x": list(range(256)),
                "y": list(rgb_data[1]),
                "type": "scatter",
                "name": "G",
                "marker": {"color": "green"},
            },
            {
                "x": list(range(256)),
                "y": list(rgb_data[2]),
                "type": "scatter",
                "name": "B",
                "marker": {"color": "blue"},
            },
        ],
        "layout": {"title": "RGB Channels"},
    }


def plot_head_pose(head_pose):
    """
    Plot Head Pose Angles.
    """
    return {
        "data": [
            {
                "x": list(range(256)),
                "y": list(head_pose[0]),
                "type": "scatter",
                "name": "pitch (x)",
            },
            {
                "x": list(range(256)),
                "y": list(head_pose[1]),
                "type": "scatter",
                "name": "yaw (y)",
            },
            {
                "x": list(range(256)),
                "y": list(head_pose[2]),
                "type": "scatter",
                "name": "roll (z)",
            },
        ],
        "layout": {"title": "3D Head Pose angles"},
    }


def plot_rPPG_signal_and_noise(rPPG, rPPG_filtered):
    """
    Plot rPPG signal.
    """
    return {
        "data": [
            # {
            #    "x": list(range(256)),
            #    "y": list(rPPG),
            #    "type": "scatter",
            #    "name": "rPPG",
            # },
            {
                "x": list(range(256)),
                "y": list(rPPG_filtered),
                "type": "scatter",
                "name": "Processed rPPG",
            },
        ],
        "layout": {"title": "rPPG signal"},
    }
