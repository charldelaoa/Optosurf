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
    x(np): linspace for the gaussian plot
    y(np): gaussian values
    """
    # 1. Define linear degrees vector and calculate Super-Gaussian
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-((x-mu)/sigma)**n)
    
    # 2. Plot 
    TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
    p = figure(title="Super-Gaussian", x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
        width = 1000, height = 550)
    p.line(x, y, line_width=4, alpha = 0.5)
    p = plot_format(p, "Degrees", "Intensity", "bottom_left", "10pt", "10pt", "10pt")
    return p, x, y


# Create a function to do the window integration
def window_integration(number_windows, window_size, x, y, p):
    """
    Performs a window integration

    Parameters
    ----------
    number_windows (int): Number of integration windows
    window_size (int): Number of data points in the window
    x(np): linspace for the gaussian plot
    y(np): gaussian values
    Returns
    -------
    p (bokeh plot): Plot of the integration
    """
    integration_points = []
    integration_axis = []
    color_multiplier = len(bokeh.palettes.Viridis256)//number_windows
    count = 0
    
    for i in range(number_windows):
    # 1. Get data in every window and integrate
        a = i*window_size
        b = i*window_size + window_size
        
        x_temp = x[a:b-gap:1]
        y_temp = y[a:b-gap:1]
        integration = np.trapz(y_temp, x_temp)
        integration_points.append(integration)

        axis = x_temp[len(x_temp)//2]
        integration_axis.append(axis)

        # 2. Draw a rectangle of the integration window
        left_edge = x_temp[0]
        right_edge = x_temp[-1]
        p.rect(x=(left_edge + right_edge)/2, y=0.18, width=right_edge-left_edge, height=0.3, fill_alpha=0.001, fill_color='green', color='green')
        p.rect(x=(right_edge + x[b-1])/2, y=0.18, width=x[b-1]-right_edge, height=0.3, fill_alpha=0.005, fill_color='red', color = 'red')
        p.circle(x_temp[::15], y_temp[::15], size = 4, color = bokeh.palettes.Viridis256[count*color_multiplier], alpha = 1)
        count += 1
    p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0')
    p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 0.8)
    return p

# 1. Define the default values for the slider variables
st.title("Super-Gaussian Equation Plotter plots")
st.title("\n $y = e^{-((x-\mu)/\sigma)^n}$")

# a. Gaussian parameters
expander_g = st.sidebar.expander("Gaussian parameters", expanded = True)
with expander_g:
    mu = st.slider("Mean", -15.0, 15.0, 0.0, 0.1)
    sigma = st.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
    n = st.slider("Order", 2, 10, 2, 2)
    number_points = st.slider("Number of points", 0, 100000, 50000, 500)
    degrees = st.slider("Select degrees range", -30.0, 30.0, (-15.0, 15.0))

# b. Integration parameters
expander_i = st.sidebar.expander("Integration parameters", expanded = True)
with expander_i:
    number_windows = st.slider("Number of windows", 1, 100, 32, 1)
    gap = st.slider("Number of gap points", 0, 1000, 100, 1)
    window_size = number_points//number_windows
    st.write('Window size: ', window_size)

# c. Report parameters
st.sidebar.title("Report parameters")


# 2. Plot gaussian equation
p, x, y = plot_equation(mu, sigma, n, number_points, degrees)


# 3. Compute window integration
p = window_integration(number_windows, window_size, x, y, p)
st.bokeh_chart(p)


# 4. Plot for different mean values
# a. Define np linear arrays for mean and standard deviation
mean_points = st.sidebar.number_input("Number of points for mu value: ", 1, 50, 31)
mean_np = np.linspace(-15, 15, mean_points)
stand_dev_points = st.sidebar.number_input("Number of points for standard deviation value: ", 1, 101, 50)
stand_dev_np = np.linspace(0.1, 5, stand_dev_points)

# b. Create a meshgrid
X, Y = np.meshgrid(mean_np, stand_dev_np)
product_grid = np.empty_like(X)

# c. Iterate over grid
for i, row in enumerate(X):
    for j, mean in enumerate(row):
        stand_dev = Y[i, j]
        product_grid[i, j] = mean * stand_dev
        # st.write(f'mean: {mean}, standard deviation: {stand_dev}')
product_grid



