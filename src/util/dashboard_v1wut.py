import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import clientside_callback
import numpy as np
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd


from dash_bootstrap_templates import load_figure_template

import threading
import src.util.plot_functions as pf

# Load figure templates for both light and dark themes
load_figure_template(["yeti", "yeti_dark"])


def run_dash_app(
    control_obj: "control.Control",
    hr_monitor: "HeartRateMonitor",
    stop_event: threading.Event,
):
    settings = control_obj.settings
    delay = settings["evaluation"]["delay"]
    fps = settings["evaluation"]["fps"]
    dash_refresh_time = settings["evaluation"]["dashboard_refresh_time"]
    len_hr_min = settings["evaluation"]["len_hr_min"]
    delay_peak = control_obj.settings["heart_rate"]["start_delay_peak_detection"]
    global should_update
    should_update = True

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME],
    )
    server = app.server
    app.layout = dbc.Container(
        [
            html.Div(
                ["Heart Rate Monitor"],
                className="bg-primary text-white h2 p-2",
            ),
            # color_mode_switch,
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="hr-info"),
                        width=4,
                        className="d-flex justify-content-center align-items-center",
                    ),
                    dbc.Col(
                        dcc.Graph(id="HR-reference-plot"),
                        width=4,
                        className="d-flex justify-content-center align-items-center",
                    ),
                ],
                justify="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    daq.StopButton(
                                        id="my-stop-button-1",
                                        buttonText="Stop Recording",
                                        n_clicks=0,
                                    ),
                                    width=12,  # Full width of the inner column
                                ),
                                dbc.Col(
                                    html.P(id="stop-button-output-1"),
                                    width=12,  # Full width of the inner column
                                ),
                            ],
                            className="d-flex flex-column justify-content-center align-items-center",
                        ),
                        width=4,
                        className="d-flex justify-content-center align-items-center",
                    ),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Switch(
                                        id="switch-input",
                                        label="Use reference measurement",
                                        value=False,
                                    ),
                                    width=12,  # Full width of the inner column
                                ),
                                dbc.Col(
                                    html.P(id="switch-output"),
                                    width=12,  # Full width of the inner column
                                ),
                            ],
                            className="d-flex flex-column justify-content-center align-items-center",
                        ),
                        width=4,
                        className="d-flex justify-content-center align-items-center",
                    ),
                ],
                justify="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="hr-plot"), width=12),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="rPPG-plot"), width=12),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="correlation-plot"), width=12),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(id="correlation-plot_hrv"), width=12
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(id="hrv-plot"),
                                        width=12,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(id="post_processed_rPPG-plot"),
                                        width=12,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(id="rhythm-plot"),
                                        width=12,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="frequency-plot"), width=12),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            dcc.Interval(
                id="plots_signals",
                interval=1 * dash_refresh_time,
                n_intervals=0,
            ),
        ],
        fluid=True,
    )

    @app.callback(
        [
            Output("rPPG-plot", "figure"),
            Output("frequency-plot", "figure"),
            Output("post_processed_rPPG-plot", "figure"),
            Output("hr-info", "figure"),
            Output("hr-plot", "figure"),
            Output("HR-reference-plot", "figure"),
            Output("hrv-plot", "figure"),
            Output("rhythm-plot", "figure"),
            Output("correlation-plot", "figure"),
            Output("correlation-plot_hrv", "figure"),
        ],
        [
            Input("plots_signals", "n_intervals"),
            # Input("color-mode-switch", "value"),
        ],
    )
    def update_signal_plots(n):
        global should_update
        if not should_update:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        rPPG_filtered = control_obj.blackboard.get_bandpass_filtered()
        post_processed_rPPG = control_obj.blackboard.get_post_processed_rPPG()
        rhythmic = control_obj.blackboard.get_samples_rhythmic()

        hr = control_obj.blackboard.get_hr_plot()
        monitoring_data, offset_md = control_obj.get_monitoring_data()

        hrv = control_obj.blackboard.get_hrv_plot()

        if rPPG_filtered is None:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )  # Corrected return type

        rPPG_plot = pf.plot_rPPG_signal_and_noise(rPPG_filtered, fps=fps)
        frequency_plot = pf.plot_frequency_domain(rPPG_filtered, fs=fps)
        post_processed_plot = pf.plot_post_processed_rPPG(post_processed_rPPG, fps=fps)
        rhythmic_plot = pf.plot_rhythmic_signal(rhythmic, fps=fps)
        if len(hr) > len_hr_min:
            hr_info = pf.plot_hr_indicator(hr[-1], hr[-2])
            if monitoring_data is not None and len(monitoring_data) > delay:
                monitoring_hr = [
                    entry["heart_rate"]
                    for entry in monitoring_data
                    if "heart_rate" in entry
                ]
                monitoring_hrv = [
                    entry["heart_rate_variability"]
                    for entry in monitoring_data
                    if "heart_rate_variability" in entry
                ]

                plot_hr = pf.plot_heart_rate(
                    hr,
                    fps=fps,
                    delay=delay,
                    monitoring_hr=monitoring_hr,
                    offset_md=offset_md,
                    delay_peak=delay_peak,
                )
                plot_hr_ref = pf.plot_hr_ref_indicator(
                    hr[-1], monitoring_hr[-delay], value="plot"
                )
                plot_hrv = pf.plot_hrv(
                    hrv,
                    fps=fps,
                    delay_peak=delay_peak,
                    monitoring_hrv=monitoring_hrv,
                    offset_md=offset_md,
                    delay=delay,
                    calibration_time=settings["evaluation"]["calibration_time_ref_hrv"],
                )
                plot_correlation_hr = pf.plot_correlation_hr(
                    hr, monitoring_hr, settings
                )
                plot_correlation_hrv = pf.plot_correlation_hrv(
                    hrv, monitoring_hrv, settings
                )
            else:
                plot_hr = pf.plot_heart_rate(hr, fps=fps, delay=delay)
                plot_hr_ref = dash.no_update
                plot_hrv = pf.plot_hrv(hrv, fps=fps, delay_peak=delay_peak)
                plot_correlation_hr = dash.no_update
                plot_correlation_hrv = dash.no_update

        else:
            hr_info = dash.no_update
            plot_hr = dash.no_update
            plot_hrv = dash.no_update
            plot_correlation_hr = dash.no_update
            plot_correlation_hrv = dash.no_update

            plot_hr_ref = dash.no_update

        return (
            rPPG_plot,
            frequency_plot,
            post_processed_plot,
            hr_info,
            plot_hr,
            plot_hr_ref,
            plot_hrv,
            rhythmic_plot,
            plot_correlation_hr,
            plot_correlation_hrv,
        )

    @app.callback(
        Output("switch-output", "children"),
        [Input("switch-input", "value")],
    )
    def on_switch_change(switch_value):
        if switch_value:
            hr_monitor.start_collection()
            return "Reference Heart Rate activated."
        else:
            hr_monitor.stop_collection()
            return "Reference Heart Rate deactivated."

    @app.callback(
        Output("stop-button-output-1", "children"),
        Input("my-stop-button-1", "n_clicks"),
    )
    def update_output(n_clicks):
        if n_clicks > 0:
            stop_event.set()
            global should_update
            should_update = False
            return f"...Recording Stopped... "
        else:
            return f"...Recording..."

    # Run the Dash app
    app.run_server(debug=False, threaded=True)
