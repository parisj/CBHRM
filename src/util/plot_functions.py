import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
import numpy as np
from scipy import stats


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


def plot_rPPG_signal_and_noise(rPPG_filtered, **kwargs):
    """
    Plot rPPG signal.
    """
    fps = kwargs.get("fps")
    # Create a sequence of indices for the x-axis based on the length of the rPPG signal
    time_axis = np.arange(len(rPPG_filtered)) / fps

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


def plot_rhythmic_signal(rPPG_filtered, **kwargs):
    """
    Plot rPPG signal.
    """
    fps = kwargs.get("fps")
    # Create a sequence of indices for the x-axis based on the length of the rPPG signal
    time_axis = np.arange(len(rPPG_filtered)) / fps

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for the rPPG signal
    fig.add_trace(
        go.Scatter(x=time_axis, y=rPPG_filtered, mode="lines", name="rPPG Signal")
    )

    # Update layout
    fig.update_layout(
        title="Rhytmic Signal ",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
    )

    return fig


def plot_frequency_domain(sig, **kwargs):
    """
    Plot the frequency domain representation of the signal.

    """
    fs = kwargs.get("fs")
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
        xaxis_range=[0, 6],
    )

    return fig


def plot_post_processed_rPPG(rppg_signal, **kwargs):
    """
    Plot the post-processed rPPG signal with time in seconds on the x-axis.

    Parameters:
    rppg_signal (numpy.ndarray): The rPPG signal array.
    fps (int): Frames per second, used to convert sample numbers to time in seconds.
    """
    fps = kwargs.get("fps")
    # Calculate the time in seconds for each sample
    time_axis = np.arange(len(rppg_signal)) / fps
    plot_range_x = len(rppg_signal) / fps
    min_x = max(0, plot_range_x - 180)
    max_x = plot_range_x
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
        xaxis_range=[min_x, max_x],
    )

    return fig


def plot_hr_indicator(hr, hr_old):
    hr = hr
    hr_old = hr_old
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=hr,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"size": 18}},  # Adjust number font size
            title={"text": "Estimated HR ", "font": {"size": 28}},
        )
    )
    fig.update_layout(
        width=250,  # Width of the figure in pixels
        height=150,  # Height of the figure in pixels
    )
    return fig


def plot_hr_ref_indicator(hr_est, hr_ref, value):
    fig = go.Figure()
    if value == "plot":
        hr_e = hr_est
        hr_r = hr_ref
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=hr_r,
                domain={"x": [0, 1], "y": [0, 1]},
                delta={
                    "reference": hr_e,
                    "relative": False,
                    "font": {"size": 15},
                },  # Adjust delta font size
                number={"font": {"size": 18}},  # Adjust number font size
                title={
                    "text": "Reference HR",
                    "font": {"size": 24},
                },
            )
        )
    fig.update_layout(
        width=250,  # Width of the figure in pixels
        height=150,  # Height of the figure in pixels
    )
    return fig


def plot_heart_rate(heartrate_data, **kwargs):
    # 120 ----->
    # 300 <-----
    """
    Plot Heart Rate data.

    Parameters:
    heartrate_data (list): Heart rate data.
    **kwargs: Optional arguments that can be 'monitoring_hr' and 'offset_md'.
    """
    fps = kwargs.get("fps")
    delay = kwargs.get("delay")
    monitored_data = kwargs.get("monitoring_hr")
    offset_md = kwargs.get("offset_md")
    delay_peak = kwargs.get("delay_peak")

    plot_range_x = len(heartrate_data) / fps
    min_x = max(0, plot_range_x - 180)
    max_x = plot_range_x

    data = [
        {
            "x": [x / fps for x in range(len(heartrate_data))],
            "y": list(heartrate_data),
            "type": "scatter",
            "name": "Estimated HR",
            "marker": {"color": "red"},
        }
    ]

    if monitored_data is not None and offset_md is not None:
        monitored_data = list(monitored_data)

        data.append(
            {
                "x": [
                    x / fps
                    for x in range(
                        offset_md,
                        offset_md + len(monitored_data),
                    )
                ],
                "y": monitored_data,
                "type": "scatter",
                "name": "Reference HR",
                "marker": {"color": "black"},
            }
        )

    return {
        "data": data,
        "layout": {
            "title": "Heart Rate",
            "xaxis": {"title": "Time (seconds)", "range": [min_x, max_x]},
            "yaxis": {"title": "Heart Rate (BPM)"},
        },
    }


def plot_hrv(hrv, **kwargs):
    # 100 <-----
    """
    Plot Heart Rate Variability (HRV).

    Parameters:
    hrv (list): HRV data.
    **kwargs: Optional arguments that can be 'monitoring_hrv' and 'offset_md'.
    """
    fps = kwargs.get("fps")
    delay_peak = kwargs.get("delay_peak")
    monitored_hrv = kwargs.get("monitoring_hrv")
    offset_md = kwargs.get("offset_md")
    delay = kwargs.get("delay")
    calibration_time_ref_hrv = kwargs.get("calibration_time")
    plot_range_x = len(hrv) / fps
    min_x = max(0, plot_range_x - 180)
    max_x = plot_range_x
    data = [
        {
            "x": [x / fps for x in range(len(hrv))],
            "y": list(hrv),
            "type": "scatter",
            "name": "Estimated HRV (RMSSD)",
            "marker": {"color": "red"},
        }
    ]

    if monitored_hrv is not None and offset_md is not None:
        monitored_hrv = list(monitored_hrv)[calibration_time_ref_hrv:]
        data.append(
            {
                "x": [
                    x / fps
                    for x in range(
                        offset_md + delay // 2 + calibration_time_ref_hrv,
                        offset_md
                        + delay // 2
                        + calibration_time_ref_hrv
                        + len(monitored_hrv),
                    )
                ],
                "y": monitored_hrv,
                "type": "scatter",
                "name": "Reference HRV",
                "marker": {"color": "black"},
            }
        )

    return {
        "data": data,
        "layout": {
            "title": "Heart Rate Variability (RMSSD)",
            "xaxis": {"title": "Time (seconds)", "range": [min_x, max_x]},
            "yaxis": {"title": "HRV RMSSD (ms)"},
        },
    }


def plot_correlation_hr(hr, hr_ref, settings):
    # 120 ---->
    # 300 total length too much
    """
    Create a correlation plot using Plotly.

    Args:
    x (np.array): Array of x values.
    y (np.array): Array of y values.

    Returns:
    plotly.graph_objs.Figure: Plotly figure object.
    """
    delay = settings["evaluation"]["delay"]
    min_ref_measurements = settings["evaluation"]["min_ref_measurements"]
    calibration_time_ref = settings["evaluation"]["calibration_time_ref"]
    start_peak_detection = settings["heart_rate"]["start_delay_peak_detection"]
    # Create Plotly figure
    fig = go.Figure()
    x = np.array(hr_ref.copy(), dtype=float)

    if x.size > min_ref_measurements:
        x = x[calibration_time_ref:-delay]
        # 200 because the reference signal has to calibrate first

        y = np.array(hr.copy())[-len(x) :]

        if x.size > y.size:
            y = y[-len(x) + calibration_time_ref :]
            x = x[-len(y) :]

        # x = x[calibration_time_ref:]
        x = x[~np.isnan(y)]  # Find the index of the first non-None value
        y = y[~np.isnan(y)]
        # Slice the array from the first non-None value to the end

        # Add trace for the rPPG signal
        stats = calculate_statistics(x, y)

        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color="#31259c", size=6, opacity=0.2, symbol="diamond-open"
                ),
                showlegend=False,
            )
        )
        x_trace = np.arange(min(x) - 20, max(x) + 20, 1)
        line_y = stats["B"] * x_trace + stats["intercept"]
        # Add linear fit line
        fig.add_trace(go.Scatter(x=x_trace, y=line_y, mode="lines", name="Fit"))

        # Add annotations for statistics
        fig.add_annotation(
            text=f"Pearson r: {stats['r']:.2f}<br>Slope B: {stats['B']:.2f}<br>Std: {stats['Std']:.2f}<br>RMSE: {stats['E']:.2f} <br>MAE: {stats['MAE']:.2f}",
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
        )

        # Update layout
        fig.update_layout(
            title="Correlation Plot HR",
            xaxis_title="Estimated Heart Rate (bpm)",
            yaxis_title="Reference Heart Rate (bpm)",
            height=600,
            width=600,
        )

    return fig


def calculate_statistics(x, y):
    """
    Calculate Pearson correlation, slope of linear fit, standard deviation, and RMSE.

    Args:
    x (np.array): Array of x values.
    y (np.array): Array of y values.

    Returns:
    dict: Dictionary containing Pearson r, slope B, standard deviation SD, and RMSE E.
    """
    x = np.array(x)
    y = np.array(y)
    x = x[-len(y) :]
    # Pearson correlation
    r, intercept = stats.pearsonr(x, y)

    # Linear fit (slope and intercept)
    slope, intercept = np.polyfit(x, y, 1)
    mae = np.mean(np.abs(np.array(y) - np.array(x)))

    # Standard deviation of y
    sd = np.std(y)

    # RMSE
    rmse = np.sqrt(np.mean((y - x) ** 2))

    return {
        "r": r,
        "B": slope,
        "Std": sd,
        "E": rmse,
        "intercept": intercept,
        "MAE": mae,
    }


def plot_correlation_hrv(hrv, hrv_ref, settings):
    """
    Create a correlation plot using Plotly.

    Args:
    x (np.array): Array of x values.
    y (np.array): Array of y values.

    Returns:
    plotly.graph_objs.Figure: Plotly figure object.
    """
    min_ref_measurements = settings["evaluation"]["min_ref_hrv"]
    calibration_time_ref = settings["evaluation"]["calibration_time_ref_hrv"]
    peak_delay = settings["heart_rate"]["start_delay_peak_detection"]
    delay = settings["evaluation"]["delay"]
    # Create Plotly figure
    fig = go.Figure()
    x = np.array(hrv_ref.copy(), dtype=float)

    if x.size > min_ref_measurements:
        x = x[calibration_time_ref : -delay // 2]

        # 200 because the reference signal has to calibrate first
        y = np.array(hrv.copy())[-len(x) :]
        if x.size > y.size:
            y = y[-len(x) + calibration_time_ref * 2 :]
            x = x[-len(y) :]
        # x = x[calibration_time_ref:]
        x = x[~np.isnan(y)]  # Find the index of the first non-None value
        y = y[~np.isnan(y)]
        # Slice the array from the first non-None value to the end

        # Add trace for the rPPG signal
        stats = calculate_statistics(x, y)

        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color="#31259c", size=6, opacity=0.2, symbol="diamond-open"
                ),
                showlegend=False,
            )
        )
        x_trace = np.arange(min(x) - 20, max(x) + 20, 1)

        line_y = stats["B"] * x_trace + stats["intercept"]
        # Add linear fit line
        fig.add_trace(go.Scatter(x=x_trace, y=line_y, mode="lines", name="Fit"))

        # Add annotations for statistics
        fig.add_annotation(
            text=f"Pearson r: {stats['r']:.2f}<br>Slope B: {stats['B']:.2f}<br>Std: {stats['Std']:.2f}<br>RMSE: {stats['E']:.2f} <br>MAE: {stats['MAE']:.2f}",
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
        )

        # Update layout
        fig.update_layout(
            title="Correlation Plot HRV",
            xaxis_title="Estimated Heart Rate Variability (ms)",
            yaxis_title="Reference Heart Rate Variability (ms)",
            height=600,
            width=600,
        )

    return fig
