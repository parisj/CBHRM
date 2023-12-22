import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_daq as daq
import dash_bootstrap_components as dbc
import time
import numpy as np
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

    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f0f0f0",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.

    side_panel = html.Div(
        [
            html.H4("Select Plots"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-hr-plot", value=True), width="auto"
                    ),
                    dbc.Col(html.Label("Heart Rate"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-hrv-plot", value=True), width="auto"
                    ),
                    dbc.Col(html.Label("Heart Rate Variability"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-hr-info", value=False), width="auto"
                    ),
                    dbc.Col(html.Label("Heart Rate Info"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-hr-reference-plot", value=False),
                        width="auto",
                    ),
                    dbc.Col(html.Label("Heart Rate Reference Info"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-rhythm-plot", value=False),
                        width="auto",
                    ),
                    dbc.Col(html.Label("Rhythmic Suppressed"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-frequency-plot", value=False),
                        width="auto",
                    ),
                    dbc.Col(html.Label("Frequency Domain"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-rppg-plot", value=False),
                        width="auto",
                    ),
                    dbc.Col(html.Label("rPPG Signal"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(
                            id="checkbox-post-processed-rppg-plot", value=False
                        ),
                        width="auto",
                    ),
                    dbc.Col(html.Label("Final rPPG Signal"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-correlation-plot", value=False),
                        width="auto",
                    ),
                    dbc.Col(html.Label("Correlation HR"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(id="checkbox-correlation-plot-hrv", value=False),
                        width="auto",
                    ),
                    dbc.Col(html.Label("Correlation HRV"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Switch(
                id="switch-input",
                label="Use Reference Device",
                label_style={"fontWeight": "bold"},  # Make label text bold
                value=False,
                style={"fontSize": 18},
            ),
            html.P(id="switch-output", style={"display": "none"}),
            dbc.Row(
                [
                    dbc.Col(
                        daq.StopButton(
                            id="my-stop-button-1",
                            buttonText="Stop Recording",
                            n_clicks=0,
                        ),
                        width=12,
                        className="d-flex justify-content-center",
                    ),
                    dbc.Col(
                        html.P(id="stop-button-output-1"),
                        width=12,
                        className="d-flex justify-content-center",
                    ),
                ],
                className="mb-2",
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    plot_area = html.Div(
        [
            html.H1("Heart Rate Monitoring Dashboard"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(dcc.Graph(id="hr-info"), id="container-hr-info"),
                            html.Div(dcc.Graph(id="hr-plot"), id="container-hr-plot"),
                            html.Div(
                                dcc.Graph(id="rhythm-plot"), id="container-rhythm-plot"
                            ),
                            html.Div(
                                dcc.Graph(id="frequency-plot"),
                                id="container-frequency-plot",
                            ),
                            html.Div(
                                dcc.Graph(id="correlation-plot"),
                                id="container-correlation-plot",
                            ),
                            # ... add more plots for the first column as needed ...
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                dcc.Graph(id="hr-reference-plot"),
                                id="container-hr-reference-plot",
                            ),
                            html.Div(dcc.Graph(id="hrv-plot"), id="container-hrv-plot"),
                            html.Div(
                                dcc.Graph(id="rPPG-plot"), id="container-rppg-plot"
                            ),
                            html.Div(
                                dcc.Graph(id="post-processed-rppg-plot"),
                                id="container-post-processed-rppg-plot",
                            ),
                            html.Div(
                                dcc.Graph(id="correlation-plot-hrv"),
                                id="container-correlation-plot-hrv",
                            ),
                            # ... add more plots for the second column as needed ...
                        ],
                        width=6,
                    ),
                ],
            ),
        ],
    )

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        side_panel,
                        width=2,  # Adjust the width of the side panel as needed
                        className="border-right",
                    ),
                    dbc.Col(
                        plot_area,
                        width=10,  # Adjust the width of the plot area as needed
                    ),
                ],
                className="g-0",  # Removes the gap between columns
            ),
            dcc.Interval(
                id="plots_signals",
                interval=1 * dash_refresh_time,  # Replace with your actual refresh time
                n_intervals=0,
            ),
            # ... other components ...
        ],
        fluid=True,
    )

    @app.callback(
        [
            Output("container-hr-plot", "style"),
            Output("container-rppg-plot", "style"),
            Output("container-frequency-plot", "style"),
            Output("container-post-processed-rppg-plot", "style"),
            Output("container-hr-info", "style"),
            Output("container-hr-reference-plot", "style"),
            Output("container-hrv-plot", "style"),
            Output("container-rhythm-plot", "style"),
            Output("container-correlation-plot", "style"),
            Output("container-correlation-plot-hrv", "style"),
        ],
        [
            Input("checkbox-hr-plot", "value"),
            Input("checkbox-rppg-plot", "value"),
            Input("checkbox-frequency-plot", "value"),
            Input("checkbox-post-processed-rppg-plot", "value"),
            Input("checkbox-hr-info", "value"),
            Input("checkbox-hr-reference-plot", "value"),
            Input("checkbox-hrv-plot", "value"),
            Input("checkbox-rhythm-plot", "value"),
            Input("checkbox-correlation-plot", "value"),
            Input("checkbox-correlation-plot-hrv", "value"),
        ],
    )
    def toggle_plot_visibility(
        hr_plot,
        rppg_plot,
        frequency_plot,
        post_processed_rppg_plot,
        hr_info,
        hr_reference_plot,
        hrv_plot,
        rhythm_plot,
        correlation_plot,
        correlation_plot_hrv,
    ):
        return (
            {"display": "block" if hr_plot else "none"},
            {"display": "block" if rppg_plot else "none"},
            {"display": "block" if frequency_plot else "none"},
            {"display": "block" if post_processed_rppg_plot else "none"},
            {"display": "block" if hr_info else "none"},
            {"display": "block" if hr_reference_plot else "none"},
            {"display": "block" if hrv_plot else "none"},
            {"display": "block" if rhythm_plot else "none"},
            {"display": "block" if correlation_plot else "none"},
            {"display": "block" if correlation_plot_hrv else "none"},
        )

    @app.callback(
        [
            Output("rPPG-plot", "figure"),
            Output("frequency-plot", "figure"),
            Output("post-processed-rppg-plot", "figure"),
            Output("hr-info", "figure"),
            Output("hr-plot", "figure"),
            Output("hr-reference-plot", "figure"),
            Output("hrv-plot", "figure"),
            Output("rhythm-plot", "figure"),
            Output("correlation-plot", "figure"),
            Output("correlation-plot-hrv", "figure"),
        ],
        [
            Input("plots_signals", "n_intervals"),
            # Input("color-mode-switch", "value"),
        ],
    )
    def update_signal_plots(n):
        current_time = time.time()

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
        samples_rgb = control_obj.blackboard.get_samples_rgb()
        samples_head_pose = control_obj.blackboard.get_samples_head_pose()
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
