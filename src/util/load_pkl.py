import pickle
import plotly.graph_objects as go
import numpy as np


def load_pickle(file_name):

    with open(file_name, "rb") as file:
        return pickle.load(file)


# Load the contents of each file
rPPG_data = load_pickle("rPPG.pkl")
head_data = load_pickle("head.pkl")
colors_data = load_pickle("colors.pkl")

r, g, b = colors_data
hx, hy, hz = head_data

r = np.array(r)
g = np.array(g)
b = np.array(b)
hx = np.array(hx)
hy = np.array(hy)
hz = np.array(hz)




def create_plot(x, y, title):
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(
        title=title,
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        template="plotly_white",  # Using the 'Yeti' template
    )
    return fig


x = np.arange(0, len(r))
create_plot(x, r, "Red Channel").show()
create_plot(x, g, "Green Channel").show()
create_plot(x, b, "Blue Channel").show()
create_plot(x, hx, "Head X").show()
create_plot(x, hy, "Head Y").show()
create_plot(x, hz, "Head Z").show()
create_plot(x, rPPG_data, "rPPG Signal").show()
