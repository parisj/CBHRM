import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import clientside_callback
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import plotly.io as pio
import pandas as pd
import time
import plot_functions as pf
from dash_bootstrap_templates import load_figure_template
import dash_daq as daq
import plotly.graph_objects as go
import threading

# Load figure templates for both light and dark themes
load_figure_template(["yeti", "yeti_dark"])


def run_dash_app(
    control_obj: "control.Control",
    hr_monitor: "HeartRateMonitor",
    stop_event: threading.Event,
):
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME],
    )

    #    color_mode_switch = html.Span(
    #        [
    #            dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
    #            dbc.Switch(
    #                id="color-mode-switch",
    #                value=False,
    #                className="d-inline-block ms-1",
    #                persistence=True,
    #            ),
    #            dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
    #        ]
    #    )

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
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="hr-info"), width=12),
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
                                        dcc.Graph(id="HR-reference-plot"), width=12
                                    ),
                                ]
                            )
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Switch(
                        id="switch-input",
                        label="Use reference measurement",
                        value=False,
                    ),
                    html.P(id="switch-output"),
                ]
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
                interval=1 * 1500,
                n_intervals=0,
            ),
        ],
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
        ],
        [
            Input("plots_signals", "n_intervals"),
            # Input("color-mode-switch", "value"),
        ],
    )
    def update_signal_plots(n):
        rPPG_filtered = control_obj.blackboard.get_bandpass_filtered()
        post_processed_rPPG = control_obj.blackboard.get_post_processed_rPPG()
        rhythmic = control_obj.blackboard.get_samples_rhythmic()

        hr = control_obj.blackboard.get_hr()
        monitoring_data, offset_md = control_obj.get_monitoring_data()
        hrv = control_obj.blackboard.get_rmssd()

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
            )  # Corrected return type

        rPPG_plot = pf.plot_rPPG_signal_and_noise(rPPG_filtered)
        frequency_plot = pf.plot_frequency_domain(rPPG_filtered)
        post_processed_plot = pf.plot_post_processed_rPPG(post_processed_rPPG)
        rhythmic_plot = pf.plot_rhythmic_signal(rhythmic)
        if len(hr) >= 2:
            hr_info = pf.plot_hr(hr[-1], hr[-2])
            if monitoring_data is not None:
                monitoring_hr = [
                    entry["heart_rate"]
                    for entry in monitoring_data
                    if "heart_rate" in entry
                ]
                plot_hr = pf.plot_heart_rate(
                    hr, monitoring_hr=monitoring_hr, offset_md=offset_md
                )
                plot_hr_ref = pf.plot_hr_ref(hr[-1], monitoring_hr[-1], value="plot")
                plot_hrv = pf.plot_hrv(hrv)
            else:
                plot_hr = pf.plot_heart_rate(hr)
                plot_hr_ref = pf.plot_hr_ref(None, None, value="plot")
                plot_hrv = pf.plot_hrv(hrv)

        else:
            hr_info = dash.no_update
            plot_hr = dash.no_update
            plot_hrv = dash.no_update

        # template = pio.templates["yeti"] if value else pio.templates["yeti_dark"]
        # plots = [
        #    rPPG_plot,
        #    frequency_plot,
        #    post_processed_plot,
        #    hr_info,
        #    plot_hr,
        #    plot_hr_ref,
        #    plot_hrv,
        # ]
        # for plot in plots:
        #    plot["layout"]["template"] = template

        return (
            rPPG_plot,
            frequency_plot,
            post_processed_plot,
            hr_info,
            plot_hr,
            plot_hr_ref,
            plot_hrv,
            rhythmic_plot,
        )

    # clientside_callback(
    #    """
    #    (switchOn) => {
    #       switchOn
    #         ? document.documentElement.setAttribute('data-bs-theme', 'light')
    #         : document.documentElement.setAttribute('data-bs-theme', 'dark')
    #       return window.dash_clientside.no_update
    #    }
    #    """,
    #    Output("color-mode-switch", "id"),
    #    Input("color-mode-switch", "value"),
    # )

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

    # Run the Dash app
    app.run_server(debug=False, threaded=True)
