import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import time
import plot_functions as pf


def run_dash_app(control_obj: "control.Control"):
    """
    Run the Dash app.
    """

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Layout of the Dash app
    app.layout = html.Div(
        [
            dcc.Graph(id="frame-plot"),
            dcc.Graph(id="roi-plot"),
            dcc.Graph(id="rgb-plot"),
            dcc.Graph(id="head-pose-plot"),
            dcc.Graph(id="rPPG-plot"),
            dcc.Graph(id="frequency-plot"),
            dcc.Graph(id="post_processed_rPPG-plot"),
            dcc.Graph(id="hr-info"),
            dcc.Interval(
                id="plot_images",
                interval=1 * 5000,
                n_intervals=0,
            ),
            dcc.Interval(
                id="plots_signals",
                interval=1 * 1000,
                n_intervals=0,
            ),
        ]
    )

    @app.callback(
        [
            Output("frame-plot", "figure"),
            Output("roi-plot", "figure"),
        ],
        [Input("plot_images", "n_intervals")],
    )
    def update_frame_plots(n):
        # print("Updating plots")
        frame, roi, rgb, head_pose = control_obj.get_samples()

        if frame is None or roi is None or rgb is None or head_pose is None:
            return (
                dash.no_update,
                dash.no_update,
            )
        frame = frame[:, :, [2, 1, 0]]
        roi = roi[:, :, [2, 1, 0]]
        frame_plot = pf.plot_frame(frame)
        roi_plot = pf.plot_roi(roi)

        return frame_plot, roi_plot

    @app.callback(
        [
            Output("rgb-plot", "figure"),
            Output("head-pose-plot", "figure"),
            Output("rPPG-plot", "figure"),
            Output("frequency-plot", "figure"),
            Output("post_processed_rPPG-plot", "figure"),
            Output("hr-info", "figure"),
        ],
        [Input("plots_signals", "n_intervals")],
    )
    def update_signal_plots(n):
        rgb = control_obj.get_samples_rgb()
        head_pose = control_obj.blackboard.get_samples_head_pose()
        rPPG = control_obj.get_samples_rPPG()
        rhythmic = control_obj.blackboard.get_samples_rhythmic()
        rPPG_filtered = control_obj.blackboard.get_bandpass_filtered()
        post_processed_rPPG = control_obj.blackboard.get_post_processed_rPPG()
        hr = control_obj.blackboard.get_hr()
        if rgb is None or head_pose is None or rPPG is None or rPPG_filtered is None:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )  # Corrected return type
        rPPG = rPPG.tolist()
        rPPG_filtered_l = rPPG_filtered.tolist()
        rgb_plot = pf.plot_rgb_channels(rgb)
        head_pose_plot = pf.plot_head_pose(head_pose)
        rPPG_plot = pf.plot_rPPG_signal_and_noise(rPPG, rPPG_filtered_l, rhythmic)
        frequency_plot = pf.plot_frequency_domain(rPPG_filtered)
        post_processed_plot = pf.plot_post_processed_rPPG(post_processed_rPPG)
        if len(hr) == 2:
            hr_info = pf.plot_hr(hr[1], hr[0])
        else:
            hr_info = dash.no_update

        return (
            rgb_plot,
            head_pose_plot,
            rPPG_plot,
            frequency_plot,
            post_processed_plot,
            hr_info,
        )

    # Run the Dash app
    app.run_server(debug=False, threaded=True)
