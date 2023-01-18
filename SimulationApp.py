import streamlit as st
import bokeh
from bokeh.plotting import figure, show
from bokeh.models import Rect
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
    # 1. Define linear degrees vector and calculate Super-Gaussian
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-((x-mu)/sigma)**n)
    

    # 2. Plot 
    TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
    p = figure(title="Super-Gaussian", x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS)
    p.line(x, y, line_width=2)
    p.circle(x, y, size = 4)
    p = plot_format(p, "Degrees", "Intensity", "bottom_left", "10pt", "10pt", "10pt")
    return p, x, y

# 1. Define the default values for the variables
st.title("Super-Gaussian Equation Plotter")
st.title("\n $y = e^{-((x-\mu)/\sigma)^n}$")

expander_g = st.sidebar.expander("Gaussian parameters", expanded = True)
with expander_g:
    mu = st.slider("Mean", -15.0, 15.0, 0.0, 0.1)
    sigma = st.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
    n = st.slider("Order", 2, 10, 4, 1)
    number_points = st.slider("Number of points", 10, 1000, 300, 1)
    degrees = st.slider("Select degrees range", -30.0, 30.0, (-15.0, 15.0))

expander_i = st.sidebar.expander("Integration parameters", expanded = True)
with expander_i:
    number_windows = st.slider("Number of windows", 1, 15, 1, 1)
    gap = st.slider("Number of gap points", 0, 15, 5, 1)
    window_size = number_points//number_windows
    st.write('Window size: ', window_size)

# 2. Plot gaussian equation
p, x, y = plot_equation(mu, sigma, n, number_points, degrees)


# 3. Define integration window
for i in range(number_windows):
    # 1. Get data in every window and integrate
    a = i*window_size
    b = i*window_size + window_size
    x_temp = x[a:b-gap:1]
    y_temp = y[a:b-gap:1]
    integration = np.trapz(x_temp,y_temp)
    
    # 2. Draw a rectangle of the integration window
    left_edge = x_temp[0]
    right_edge = x_temp[-1]
    p.rect(x=(left_edge + right_edge)/2, y=0.25, width=right_edge-left_edge, height=0.5, fill_alpha=0.1, fill_color='green')
    p.rect(x=(right_edge + x[b-1])/2, y=0.25, width=x[b-1]-right_edge, height=0.5, fill_alpha=0.1, fill_color='red')
    
st.bokeh_chart(p)
    
    




