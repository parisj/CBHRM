import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
import numpy as np


def plot_frame(frame_data):
    """
    Plot frame data as an image.
    """
    fig = px.imshow(frame_data, binary_string=True)
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,  # Hide the axis
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,  # Hide the axis labels and ticks
        plot_bgcolor="rgba(0,0,0,0)",  # Optional: set background color to transparent
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
    )
    return fig


def plot_roi(roi_data):
    """
    Plot ROI (Region of Interest) data as an image.
    """
    fig = px.imshow(roi_data, binary_string=True)
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,  # Hide the axis
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,  # Hide the axis labels and ticks
        plot_bgcolor="rgba(0,0,0,0)",  # Optional: set background color to transparent
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
    )
    return fig


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


def plot_rPPG_signal_and_noise(rPPG_filtered):
    """
    Plot rPPG signal.
    """

    # Create a sequence of indices for the x-axis based on the length of the rPPG signal
    time_axis = np.arange(len(rPPG_filtered)) / 20

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for the rPPG signal
    fig.add_trace(
        go.Scatter(x=time_axis, y=rPPG_filtered, mode="lines", name="rPPG Signal")
    )

    # Update layout
    fig.update_layout(
        title="Rhythmic Noise Suppressed rPPG Signal",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
    )

    return fig


def plot_frequency_domain(sig, fs=30):
    """
    Plot the frequency domain representation of the signal.

    """
    # Compute the Fourier Transform of the signal
    sig_fft = np.fft.fft(sig)

    # Compute the two-sided spectrum
    two_sided_spectrum = np.abs(sig_fft) / len(sig)

    # Compute the one-sided spectrum (only positive frequencies)
    one_sided_spectrum = two_sided_spectrum[: len(sig) // 2]

    # Create frequency axis (only positive frequencies)
    freq_axis = np.fft.fftfreq(len(sig), 1 / fs)[: len(sig) // 2]

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for spectrum
    fig.add_trace(
        go.Scatter(x=freq_axis, y=one_sided_spectrum, mode="lines", name="Magnitude")
    )

    # Update layout
    fig.update_layout(
        title="Frequency domain representation",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Magnitude",
        template="plotly_white",
        xaxis_range=[0, 10],
    )

    return fig


def plot_post_processed_rPPG(rppg_signal, fps=20):
    """
    Plot the post-processed rPPG signal with time in seconds on the x-axis.

    Parameters:
    rppg_signal (numpy.ndarray): The rPPG signal array.
    fps (int): Frames per second, used to convert sample numbers to time in seconds.
    """
    # Calculate the time in seconds for each sample
    time_axis = np.arange(len(rppg_signal)) / fps

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for the rPPG signal
    fig.add_trace(
        go.Scatter(x=time_axis, y=rppg_signal, mode="lines", name="rPPG Signal")
    )

    # Update layout
    fig.update_layout(
        title="Post-Processed rPPG Signal",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
    )

    return fig


def plot_hr(hr, hr_old):
    hr = hr
    hr_old = hr_old
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=hr,
            domain={"x": [0, 0.5], "y": [0.5, 1]},
            title={"text": "Estimated HR ", "font": {"size": 24}},
        )
    )
    fig.update_layout(
        width=300,  # Width of the figure in pixels
        height=300,  # Height of the figure in pixels
    )
    return fig


def plot_hr_ref(hr_est, hr_ref, value):
    fig = go.Figure()
    if value == "plot":
        hr_e = hr_est
        hr_r = hr_ref
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=hr_r,
                domain={"x": [0, 0.5], "y": [0.5, 1]},
                delta={"reference": hr_e, "relative": False},
                title={
                    "text": "Reference HR (r)",
                    "font": {"size": 24},
                },
            )
        )
    fig.update_layout(
        width=300,  # Width of the figure in pixels
        height=300,  # Height of the figure in pixels
    )
    return fig


def plot_heart_rate(heartrate_data, **kwargs):
    """
    Plot Heart Rate data.

    Parameters:
    heartrate_data (list): Heart rate data.
    **kwargs: Optional arguments that can be 'monitoring_hr' and 'offset_md'.
    """
    monitored_data = kwargs.get("monitoring_hr")
    offset_md = kwargs.get("offset_md")

    data = [
        {
            "x": [x / 20 for x in range(len(heartrate_data))],
            "y": list(heartrate_data),
            "type": "scatter",
            "name": "Estimated HR",
            "marker": {"color": "red"},
        }
    ]

    if monitored_data is not None and offset_md is not None:
        data.append(
            {
                "x": [
                    x / 20 for x in range(offset_md, offset_md + len(monitored_data))
                ],
                "y": list(monitored_data),
                "type": "scatter",
                "name": "Reference HR",
                "marker": {"color": "black"},
            }
        )

    return {
        "data": data,
        "layout": {
            "title": "Heart Rate",
            "xaxis": {"title": "Time (seconds)"},
            "yaxis": {"title": "Heart Rate (BPM)"},
        },
    }


def plot_hrv(hrv, **kwargs):
    """
    Plot Heart Rate Variability (HRV).

    Parameters:
    hrv (list): HRV data.
    **kwargs: Optional arguments that can be 'monitoring_hrv' and 'offset_md'.
    """
    monitored_hrv = kwargs.get("monitoring_hrv")
    offset_md = kwargs.get("offset_md")

    data = [
        {
            "x": [x / 20 for x in range(len(hrv))],
            "y": list(hrv),
            "type": "scatter",
            "name": "Estimated HRV (RMSSD)",
            "marker": {"color": "red"},
        }
    ]

    if monitored_hrv is not None and offset_md is not None:
        data.append(
            {
                "x": [x / 20 for x in range(offset_md, offset_md + len(monitored_hrv))],
                "y": list(monitored_hrv),
                "type": "scatter",
                "name": "Reference HRV",
                "marker": {"color": "black"},
            }
        )

    return {
        "data": data,
        "layout": {
            "title": "Heart Rate Variability (RMSSD)",
            "xaxis": {"title": "Time (seconds)"},
            "yaxis": {"title": "HRV Value"},
        },
    }
