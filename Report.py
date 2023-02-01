# ---
# title: "Optosurf simulation"
# format: 
#   html:
#     code-fold: true
#     code-tools: true
#     theme: slate
#     toc: true
#     toc-location: left
#     number-sections: true
#     fontcolor: "#E3F4FF"
# jupyter: python3
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: WebApps
#     language: python
#     name: python3
# ---

# + vscode={"languageId": "python"}
#| echo: false
import bokeh
from bokeh.plotting import figure, curdoc, show
from bokeh.models import Rect, LinearColorMapper, SingleIntervalTicker, LinearAxis, Grid
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
import numpy as np
import altair as alt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
output_notebook()

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

    plot.background_fill_color = "#282B30"
    plot.border_fill_color = "#282B30"

    plot.xgrid.grid_line_color = '#606773'
    plot.ygrid.grid_line_color = '#606773'

    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "15pt"
    return plot
# + vscode={"languageId": "python"}

# -


# # <span style="color:#A6DDFF">Optical field definition and window integratio bb</span>

# ## Optical field definition
# The optical field detected by the Optosurf is simulated according to the equation $y = e^{-((x-\mu)/\sigma)^n}. 
# The $\mu$ parameter relates to the incoming angle of the sample's reflected light, this is equivalent to a lateral shift of the optical field. The $\sigma$ 
# parameter relates roughness of the sample and determines the widht of the optical field.

# ## Optosurf 32 pixel line detector 
# The Optosurf head has a 32 pixel linear detector. In order to simulate this, the equation of the optical field is defined. Then, a window integration over 32 windows is performed over the optical field (@fig-1 a). The outcome of the integration will be 32 sdampling points over a range of 30 degrees, from -15 to 15 (@fig-1 b). Finally, a histogram is calculated from these points (@fig-1 c).
#
#

#
# ::: {#fig-1}
#
# ![](images/Fig2.png)
#
# a. Optical field definition. b. Window integration. c. Histogram
# :::
#

# ## Optical field and window integration Python simulation
# The previous process was coded in python using three functions:
#
# -<span style="color:#EEDA89">plot-equation</span>: This function creates the optical field according to the **input parameters** $\mu$, $\sigma$, $n$, number of points and degrees range. The **output** returns two arrays: x and y, a linear array for the x scale and the corresponding calcuted y values
#
# -<span style="color:#EEDA89">window-integration</span>: This function performs the window integration simulating the linear detector. The **input parameters** are number of windows, the window size in points and the number of gap points between each pixel. The **output** are the new integration axis and the 32 sampling points.
#
# -<span style="color:#EEDA89">histogram_reconstruction</span>: The final function takes the new axis and the sampled points and reconstructs the histogram. 
#
# The code is now shown:

# ### Functions definition

# + vscode={"languageId": "python"}
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
    ticker = SingleIntervalTicker(interval=2.5, num_minor_ticks=10)
    xaxis = LinearAxis(ticker = ticker)
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-abs(((x-mu)/sigma))**n)
    
    # 2. Plot 
    if plot:
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = width, height = height)
        p.line(x, y, line_width=4, alpha = 0.5, line_color = "#C5E064")
        p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        p = plot_format(p, "Degrees", "Intensity", "bottom_left", "10pt", "10pt", "10pt")
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
    color_multiplier = len(bokeh.palettes.Viridis256)//number_windows
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

        # 2. Draw a rectangle of the integration window
        if p is not None:
            left_edge = x_temp[0]
            right_edge = x_temp[-1]
            p.rect(x=(left_edge + right_edge)/2, y=0.18, width=right_edge-left_edge, height=0.3, fill_alpha=0.001, fill_color='#C5E0B4', color='#C5E0B4')
            p.rect(x=(right_edge + x[b-1])/2, y=0.18, width=x[b-1]-right_edge, height=0.3, fill_alpha=0.005, fill_color='#F16C08', color = '#F16C08')
            p.circle(x_temp[::15], y_temp[::15], size = 4, alpha = 1)
            count += 1
    if p is not None:
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0')
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 0.8)
    return p, integration_axis, integration_points


def histogram_reconstruction(int_points, hist_bool, title="Super-Gaussian", width = 700, height = 550):
    """
    Constructs a histrogram

    Parameters
    ----------
    int_points(np): Points calculated from the window integration
    Returns
    -------
    hist_plot (bokeh plot): Plot of the histogram
    std_dev(float): Histogram's standard deviation
    """
    
    # a. Histogram reconstruction
    normalized_y = np.multiply(int_points, 10000)
    hist_2d = np.array([])
    for i, int_point in enumerate(normalized_y):
        round_int_point = round(float(int_point))
        hist_2d = np.concatenate((hist_2d, np.array([float(i)]*round_int_point)))
    
    # b. Calculate standard deviation
    stddev = np.std(hist_2d)
    
    # c. Plot histogram
    if hist_bool:
        title = f"Std_dev: {stddev:.5f}; " + title
        hist_plot = figure(title = title, plot_width=width, plot_height=height, x_axis_label='x', y_axis_label='count()')
        hist, edges = np.histogram(hist_2d, bins=20)
        hist_plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='#2D908C', line_color='black', alpha=0.7)
        hist_plot = plot_format(hist_plot, "x", "Count", "bottom_left", "10pt", "10pt", "10pt")
        return hist_plot, np.std(stddev)

    else:
        return stddev


# -

# ### Optical field parameter sweep

# Once the functions have been defined a parameter sweep is made. This includes sweeping the $\mu$ and $\sigma$ parameters of the optical field equation. In practice,
# sweeping $\mu$ parameter is equivalent to changing the angle of the incoming light which produce a linear shift in the optical field as illustrated in @fig-2 a.
#
#  The $\sigma$ parameter broaded the width of the Gaussian field and is equivalent to samples with different reflectivities as shown in @fig-2 b.
#
#

# ::: {#fig-2}
#
# ![](images/Fig2.png)
#
# a. Sweep in mu parameter. b. Sweep in sigma parameter.
# :::

# ### Sweep parameter simulation
# The sweep parameter was simulated using the previously defined functions. The sweep parameters are:
#
# -<span style="color:#EEDA89">$\mu$</span>: from -4 to -4 degrees
#
# -<span style="color:#EEDA89">$\sigma$</span>: from 1 to 2.5
#
# For each sweep combination the histogram is reconstructed and its standard deviation is stored as a matrix. The code and output plot (notice plot is interactive) are now shown:

# + vscode={"languageId": "python"}
#| column: screen
# 1. Define input parameters for the optical field
mu_np = np.linspace(-4, 4, 3)
std_np = np.linspace(1.0, 2.5, 3)
X, Y = np.meshgrid(mu_np, std_np)
n = 3.5
number_points = 10000
degrees = [-15, 15]
number_windows = 32
window_size = number_points//number_windows
gap = 100

# 2. Define empty plot lists
plots_gaussian = []
plots_histogram = []

for i in range(len(mu_np)):
    for j in range(len(std_np)):
        # a. Calculate window integration
        title = f"mu: {mu_np[i]:.3f}, sigma: {std_np[j]:.3f}"
        plot, x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, True, title, 320, 200)
        plot.title.text_font_size = "10pt"
        plot, int_axis, int_points = window_integration(number_windows, window_size, gap, x, y, plot)
        plots_gaussian.append(plot)

        # b. Calculate histogram
        hist_title = f"mu: {mu_np[i]:.3f}, sigma: {std_np[j]:.3f}"
        hist_plot, std_dev = histogram_reconstruction(int_points, True, hist_title, 320, 200)
        hist_plot.title.text_font_size = "8pt"
        plots_gaussian.append(hist_plot)

grid_gaussian = gridplot(children = plots_gaussian, ncols = 6, merge_tools=False)
show(grid_gaussian)

# -

# Notice that the simulation confirmed the expected behaviour. The sweep in $\mu$ is equivalent to a linear shift a long the x-axis, while the sweep in $\sigma$ broaded the optical field, hence obtaining more sampling points inside the curve.

# ## Standard deviation matrix
# Finally, a new sweep is performed with the parameters: 
#
# -<span style="color:#EEDA89">$\mu$</span>: from -5 to -5 degrees
#
# -<span style="color:#EEDA89">$\sigma$</span>: from 1 to 2.5
#

# + vscode={"languageId": "python"}
#| column: column-page
number_points = 5000
number_windows = 32
window_size = number_points//number_windows
gap = 100
n = 3.5
degrees = [-15, 15]
           
mu_range = [-5, 5]
axis_range = [mu_range[0], mu_range[1], 0.5]
axis_range = np.arange(-5, 5.5, 0.5)
mu_step = 0.10
mu_points = (mu_range[1] - mu_range[0])//mu_step

std_range = [1.5, 1.6]
std_step = 0.1 
std_points = (std_range[1] - std_range[0])/std_step


mu_np = np.linspace(mu_range[0], mu_range[1], int(mu_points))
std_np = np.linspace(std_range[0], std_range[1], int(std_points))
X, Y = np.meshgrid(mu_np, std_np)
std_grid = np.empty_like(X)

# Iterates mu and standard deviation
for i in range(len(mu_np)):
    for j in range(len(std_np)):
        # Generate x and y Gaussian data points 
        title = f"mu: {mu_np[i]:.3f}, std: {std_np[j]:.3f}"
        # print(title)
        x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, False)
        p, int_axis, int_points = window_integration(number_windows, window_size, gap, x, y)
        std_grid[j, i] = histogram_reconstruction(int_points, False)
        
source = pd.DataFrame({'mu': X.ravel(),
                        'std': Y.ravel(),
                        'value': std_grid.ravel()})

intensity_plot = alt.Chart(source).mark_rect().encode(
    alt.X('mu:O', axis=alt.Axis(values=axis_range, format=".1f")),
    y='std:O',
    color='value:Q',
    tooltip='value').interactive()
intensity_plot = intensity_plot.properties(width = 600, height = 200, title = 'Standard deviation variation')
intensity_plot
# -

# Standard deviation: 1.0

# + vscode={"languageId": "python"}
intensity_plot
# -

# Standard deviation: 1.5

# + vscode={"languageId": "python"}
intensity_plot

# + vscode={"languageId": "python"}
# https://towardsdatascience.com/version-control-with-jupyter-notebook-b9630bc5996e
