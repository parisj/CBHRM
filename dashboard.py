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
            dcc.Interval(
                id="interval-update",
                interval=1 * 105,
                n_intervals=0,
            ),
        ]
    )

    @app.callback(
        [
            Output("frame-plot", "figure"),
            Output("roi-plot", "figure"),
            Output("rgb-plot", "figure"),
            Output("head-pose-plot", "figure"),
        ],
        [Input("interval-update", "n_intervals")],
    )
    def update_plots(n):
        # print("Updating plots")
        frame, roi, rgb, head_pose = control_obj.get_samples()

        if frame is None or roi is None or rgb is None or head_pose is None:
            return dash.no_update  # Don't update the figure if data is None
        frame = frame[:, :, [2, 1, 0]]
        roi = roi[:, :, [2, 1, 0]]
        frame_plot = pf.plot_frame(frame)
        roi_plot = pf.plot_roi(roi)
        rgb_plot = pf.plot_rgb_channels(rgb)
        head_pose_plot = pf.plot_head_pose(head_pose)
        return frame_plot, roi_plot, rgb_plot, head_pose_plot

    # Run the Dash app
    app.run_server(debug=False, threaded=True)
