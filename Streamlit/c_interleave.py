import streamlit as st
from bokeh.plotting import figure
from bokeh.models import Range1d, Span
from bokeh.layouts import gridplot
import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, BSpline


st.set_page_config(page_title="Interleave", layout="wide")


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
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"
    # plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.0

    # Title format
    plot.title.text_font_size = titlesize
    plot.background_fill_color = "#0E1117"
    plot.border_fill_color = "#0E1117"

    plot.xgrid.grid_line_color = '#2D3135'
    plot.ygrid.grid_line_color = '#2D3135'
    
    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "8pt"

    plot.legend.click_policy="hide"
    return plot

def plot_equation(mu, sigma, n, number_points, degrees, plot, title="Super-Gaussian", width = 700, height = 550):
    """
    Plot the optical field equation using Bokeh

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
    # ticker = SingleIntervalTicker(interval=1, num_minor_ticks=1)
    # xaxis = LinearAxis(ticker = ticker)
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-abs(((x-mu)/sigma))**n)
    
    # 2. Plot 
    if plot:
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = width, height = height, x_range=Range1d(-5, 5), y_range=Range1d(-0.5, 1.2))
        vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=2)
        p.add_layout(vline)
        # p.line(x[::20], y[::20], line_width=4, alpha = 0.5, line_color = "#C5E064")
        # p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        # p.add_layout(xaxis, 'below')
        return p, x, y
    else:
        return x, y
    

def window_integration(number_windows, window_size, gap, x, y, p=None):
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
    integration_axis (np): window integration axis
    integration_points (np): Integrated points
    """
    integration_points = []
    integration_axis = []
    count = 0
    
    for i in range(number_windows):
    # 1. Get data in every window and integrate
        a = i*window_size
        b = i*window_size + window_size
        
        x_temp = x[a:b-gap:1]
        y_temp = y[a:b-gap:1]
        integration = np.trapz(y_temp, x_temp, dx = x[1] - x[0])
        integration_points.append(integration)

        axis = x_temp[len(x_temp)//2]
        integration_axis.append(axis)

    if p is not None:
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 0.8)
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0', legend_label = 'Sampled Points')
        p.x_range = Range1d(-5, 5)
        p.y_range = Range1d(-0.5, 1)
        p.xaxis.ticker.desired_num_ticks = 10
        p = plot_format(p, "Degrees", "Intensity", "top_left", "7pt", "7pt", "7pt")
    integration_axis = np.array(integration_axis)
    integration_points = np.array(integration_points)
    return p, integration_axis, integration_points


# %% 1. Define the default values for the slider variables
st.title("Optical field definition as: $y = e^{-((x-\mu)/\sigma)^n}$")
# a. Streamlit sliders -Gaussian parameters
st.sidebar.title("Gaussian parameters")
expander_g = st.sidebar.expander("Gaussian parameters", expanded = True)
with expander_g:
    st.markdown("With these sliders you can modify the parameters of $y = e^{-((x-\mu)/\sigma)^n}$")
    mu_low = st.number_input("Select the low mu", -5.0, 5.0, 0.1)
    mu_high = st.number_input("Select the high mu", -5.0, 5.0, 1.0)
    mu_points = st.number_input("Input the number of mu points", 0, 50, 10)
    mu_np = np.linspace(mu_low, mu_high, mu_points)
    mu_options = st.multiselect('Mu values', mu_np, mu_np)
    mu_options.sort()

    sigma = st.slider("Standard Deviation", 0.1, 5.0, 1.3, 0.1)
    std_range = [sigma, sigma]
    std_step = st.number_input("Input the standard deviation step", 0.0, 5.0, 1.0)
    std_points = (std_range[1] - std_range[0])/std_step
    std_np = np.linspace(std_range[0], std_range[1], int(std_points)+1)

    n = st.number_input("Order", 0.0, 10.0, 3.4)
    number_points = st.slider("Number of points", 0, 100000, 5000, 500)
    degrees = st.slider("Select degrees range", -30.0, 30.0, (-15.0, 15.0))

# b. Integration parameters
st.sidebar.title("Integration parameters")
expander_i = st.sidebar.expander("Integration parameters", expanded = False)
with expander_i:
    st.markdown("These parameters are the number of integration windows and gap points")
    number_windows = st.slider("Number of windows", 1, 100, 32, 1)
    gap = st.slider("Number of gap points", 0, 1000, 0  , 1)
    window_size = number_points//number_windows
    st.write('Window size: ', window_size)


# %% 2. Define starting plot grid and perform window integration
mu_np_a = np.array(mu_options)

plots_gaussian = []
columns = ['mu', 'int_axis', 'int_points']
df = pd.DataFrame(columns=columns)

for i in range(len(mu_np_a)):
    for j in range(len(std_np)):
        # Generate x and y Gaussian data points 
        title = f"mu: {mu_np_a[i]:.1f}, std: {std_np[j]:.3f}"
        p, x, y = plot_equation(mu_np_a[i], std_np[j], n, number_points, degrees, True, title, 320, 260)
        p, int_axis, int_points = window_integration(number_windows, window_size, gap, x, y, p)
        plots_gaussian.append(p)
        df = df.append(pd.DataFrame([[mu_np_a[i], int_axis, int_points]], columns=columns), ignore_index=True)

grid_gaussian = gridplot(children = plots_gaussian, ncols = 4, merge_tools=False)
st.bokeh_chart(grid_gaussian)
