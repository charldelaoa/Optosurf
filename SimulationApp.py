import streamlit as st
import bokeh
from bokeh.plotting import figure, show
import numpy as np

st.set_page_config(page_title="Super-Gaussian Equation Plotter", layout="wide")


def plot_format(plot, xlabel, ylabel, location, size, titlesize, labelsize):
    # x axis format
    plot.xaxis.axis_label = xlabel
    plot.xaxis.axis_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_style = "bold"
    plot.xaxis.axis_label_text_font_size = size
    plot.xaxis.major_label_text_font_size = size

    # y axis format
    plot.yaxis.axis_label = ylabel
    plot.yaxis.axis_label_text_font_style = 'bold'
    plot.yaxis.major_label_text_font_style = "bold"
    plot.yaxis.axis_label_text_font_size = size
    plot.yaxis.major_label_text_font_size = size

    # Legend format
    plot.legend.location = location
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = labelsize
    plot.legend.label_text_font_style = 'bold'
    plot.legend.border_line_width = 3
    plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.5

    # Title format
    plot.title.text_font_size = titlesize
    return plot


# Create a function to plot the equation
def plot_equation(mu, sigma, n, number_points, degrees):
    """
    Plot the Super-Gaussian equation using Bokeh

    Parameters
    ----------
    mu (float): Mean value for the equation
    sigma (float): Standard deviation value for the equation
    n (float): Order value for the equation
    number_points (int): number of points to calculate the function
    
    Returns
    -------
    plots (bokeh plot): Plot of the Super-Gaussian equation
    """
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-((x-mu)/sigma)**n)
    p = figure(title="Super-Gaussian", x_axis_label='x', y_axis_label='y')
    p.line(x, y, line_width=2)
    p.circle(x, y, size = 4)
    p = plot_format(p, "Degrees", "Intensity", "bottom_left", "10pt", "10pt", "10pt")
    st.bokeh_chart(p)



# 1. Define the default values for the variables
mu = st.sidebar.slider("Mean", -5.0, 5.0, 0.0, 0.1)
sigma = st.sidebar.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
n = st.sidebar.slider("Order", 2, 10, 4, 1)
number_points = st.sidebar.slider("Number of points", 10, 1000, 32, 1)
degrees = st.sidebar.slider("Select degrees range", -30.0, 30.0, (-15.0, 15.0))

# Create the Streamlit app
st.title("Super-Gaussian Equation Plotter")

plot_equation(mu, sigma, n, number_points, degrees)