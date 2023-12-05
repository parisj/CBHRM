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
    time_axis = np.arange(len(rPPG_filtered))

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for the rPPG signal
    fig.add_trace(
        go.Scatter(x=time_axis, y=rPPG_filtered, mode="lines", name="rPPG Signal")
    )

    # Update layout
    fig.update_layout(
        title="Rhythmic Noise Suppressed rPPG Signal",
        xaxis_title="Sample Number",
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
    )

    return fig


def plot_post_processed_rPPG(rppg_signal):
    """
    Plot the post-processed rPPG signal.

    """
    # Create a sequence of indices for the x-axis based on the length of the rPPG signal
    time_axis = np.arange(len(rppg_signal))

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for the rPPG signal
    fig.add_trace(
        go.Scatter(x=time_axis, y=rppg_signal, mode="lines", name="rPPG Signal")
    )

    # Update layout
    fig.update_layout(
        title="Post-Processed rPPG Signal",
        xaxis_title="Sample Number",
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
    Plot RGB channels.
    """
    if not kwargs:
        monitored_data = None
        offset_md = None
    else:
        monitored_data = kwargs["monitoring_hr"]
        offset_md = kwargs["offset_md"]

    if monitored_data is None or offset_md is None:
        return {
            "data": [
                {
                    "x": list(range(11230)),
                    "y": list(heartrate_data),
                    "type": "scatter",
                    "name": "R",
                    "marker": {"color": "red"},
                },
            ],
            "layout": {"title": "Hear Rate "},
        }
    else:
        return {
            "data": [
                {
                    "x": list(range(11230)),
                    "y": list(heartrate_data),
                    "type": "scatter",
                    "name": "Estimated HR",
                    "marker": {"color": "red"},
                },
                {
                    "x": list(range(offset_md, 11230)),
                    "y": list(monitored_data),
                    "type": "scatter",
                    "name": "Reference HR",
                    "marker": {"color": "black"},
                },
            ],
            "layout": {"title": "Hear Rate "},
        }


def plot_hrv(hrv, **kwargs):
    """
    Plot RGB channels.
    """
    if not kwargs:
        monitored_hrv = None
        offset_md = None
    else:
        monitored_hrv = kwargs["monitoring_hrv"]
        offset_md = kwargs["offset_md"]

    if monitored_hrv is None or offset_md is None:
        return {
            "data": [
                {
                    "x": list(range(11230)),
                    "y": list(hrv),
                    "type": "scatter",
                    "name": "R",
                    "marker": {"color": "red"},
                },
            ],
            "layout": {"title": "Hear Rate Variability (RMSSD)"},
        }
    else:
        return {
            "data": [
                {
                    "x": list(range(11230)),
                    "y": list(hrv),
                    "type": "scatter",
                    "name": "Estimated HRV (RMSSD)",
                    "marker": {"color": "red"},
                },
                {
                    "x": list(range(offset_md, 11230)),
                    "y": list(monitored_hrv),
                    "type": "scatter",
                    "name": "Reference HRV",
                    "marker": {"color": "black"},
                },
            ],
            "layout": {"title": "Hear Rate "},
        }
