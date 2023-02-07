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

# +
# from simulation_functions import plot_format, plot_equation, window_integration
# https://realpython.com/numpy-scipy-pandas-correlation-python/
from bokeh.plotting import figure, curdoc, show
from bokeh.models import Rect, SingleIntervalTicker, LinearAxis, Grid, Range1d, AdaptiveTicker, Title
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
import numpy as np
import altair as alt
import pandas as pd
import numpy as np
from math import ceil
import pickle
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
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"
    # plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.3

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



# -

# # Part 2
# * Create a supergaussian function: $y = e^{-((x-\mu)/\sigma)^n}$ with n = 3.4 and narrow $\sigma$
# * Vary the $\mu$ parameter to introduce lateral displacements
# * Perform a window integration to get 32 sampling points
# * Reconstruct the original supergaussian function based on the sampling points
#
#

# ## Define a Gaussian and sweep the $\mu$ parameter

# +
number_points = 5000
number_windows = 32
window_size = number_points//number_windows
gap = 100
n = 3.4
degrees = [-15, 15]
           
mu_range = [-1.0, 1.0]
mu_step = 0.10
mu_points = ceil((mu_range[1] - mu_range[0])/mu_step)

std_range = [1.2, 1.3]
std_step = 0.1 
std_points = (std_range[1] - std_range[0])/std_step

mu_np = np.linspace(mu_range[0], mu_range[1], int(mu_points)+1)
std_np = np.linspace(std_range[0], std_range[1], int(std_points))
plots_gaussian = []
int_points_a = []
int_axis_a = []
X, Y = np.meshgrid(mu_np, std_np)
std_grid = np.empty_like(X)
columns = ['mu', 'int_axis', 'int_points']
df = pd.DataFrame(columns=columns)

# Iterates mu and standard deviation
for i in range(len(mu_np)):
    for j in range(len(std_np)):
        # Generate x and y Gaussian data points 
        title = f"mu: {mu_np[i]:.1f}, std: {std_np[j]:.3f}"
        p, x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, True, title, 280, 260)
        # p.title.text_font_size = "10pt"
        p, int_axis, int_points = window_integration(number_windows, window_size, gap, x, y, p)
        plots_gaussian.append(p)
        df = df.append(pd.DataFrame([[mu_np[i], int_axis, int_points]], columns=columns), ignore_index=True)
df.to_csv('data/b_shifted_Gaussian.csv', index=False)


# +
# #| column: screen
# grid_gaussian = gridplot(children = plots_gaussian, ncols = 6, merge_tools=False)
# show(grid_gaussian)
# -

# ## Calculate average of sampled points and correlation coefficient

# +
# a. Get average integration axis
int_axis_values = df['int_axis'].to_numpy()
int_axis_average = np.mean(int_axis_values, axis=0)
int_axis_average

# b. Get sampled points
int_points_values = df['int_points'].to_numpy()
int_points_average = np.mean(int_points_values, axis=0)

# c. Calculate correlation coefficient between the average points and the integration points
correlation_values = []
for int_points in int_points_values:
    corr_coef = np.corrcoef(int_points_average, int_points)
    correlation_values.append(corr_coef[0,1])
    
# d. Add correlation values as a column to df
df['Correlation'] = correlation_values

# e. Create df with average integration points
df_average = pd.DataFrame({'int_axis_average': int_axis_average, 
                          'int_points_average': int_points_average
                            })

# f. Merge df's
updated_df = df.reindex(df_average.index).merge(df_average, how='left', left_index=True,
                                                right_index=True)
updated_df.to_csv('data/b_shifted_Gaussian.csv', index=False)
plots_gaussian_b = plots_gaussian.copy()
# -

for i, plot in enumerate(plots_gaussian_b):
    mu = df.loc[i, 'mu']
    corr_coef = df.loc[i, 'Correlation']
    plot.circle(int_axis_average, int_points_average, size = 7, color = '#C5E0B4', legend_label = f'Avg. points - Corr. Coeff {corr_coef:.4f}')
    plot.line(int_axis_average, int_points_average, line_width = 4, color = '#C5E0B4', alpha = 0.7)


# ## Plot sampled points and average points with correlation coefficient

#| column: screen
grid_gaussian_average = gridplot(children = plots_gaussian_b, ncols = 6, merge_tools=False)
show(grid_gaussian_average)

# ## Make spline interpolation of average points and compare to original function

# +
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, BSpline

def calculate_fwhm(x, y):
    """
    Calculates fwhm

    Parameters
    ----------
    x(np): x values
    y(np): y values
    Returns
    -------
    fwhm(float): fwhm
    """
    half_max = np.max(y) / 2
    x_left = x[np.where(y >= half_max)[0][0]]
    x_right = x[np.where(y >= half_max)[0][-1]]
    return x_right - x_left


def spline_interpolation(x, y, new_axis, width = 600, height = 450):
    """
    Calculates a spline interpolation of the sampled points 
    by the window integration 

    Parameters
    ----------
    mu_np(np): range to vary mu parameter of gaussian function
    std_np(np): range of standard deviation to vary
    Returns
    -------
    std_grid(np): standard deviation matrix
    """

    TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]

    interpolation_methods = [("Cubic spline", CubicSpline, '#C5E0B4'), 
                            ("Pchip", PchipInterpolator, '#EEDA89'), 
                            ("Akima", Akima1DInterpolator, '#FEEED9')]
    
    plots = [figure(width = width, height = height, tooltips = TOOLTIPS) for i in range(len(interpolation_methods))]
    
    for i, (method, interp_func, color) in enumerate(interpolation_methods):
        # a. Interpolate data  
        interp = interp_func(x, y)
        interp_points = interp(new_axis)
        
        # b. Calculate FWHM 
        # fwhm = calculate_fwhm(new_axis, interp_points)
        # hist_plot, std_dev = histogram_reconstruction(interp_points, True)
        
        # c. Calculate correlation coefficients
        corr_coef = np.corrcoef(y_original, interp_points)[0,1]

        # d. Plot the data
        plots[i].line(x_original, y_original, line_width=4, legend='Original Gaussian')
        plots[i].line(new_axis, interp_points, line_width=4, color= color, legend = method)
        plots[i].circle(x, y, size=7, color='#FAA0A0', legend = 'Average points')
        # plots[i].title = f"Method: {method}, FWHM: {fwhm}"
        plots[i].title = f"Method: {method}; Corr. Coefficient: {corr_coef:.4f}"
        plots[i] = plot_format(plots[i], "Degrees", "Intensity", "right", "7pt", "7pt", "7pt")
        plots[i].add_layout(plots[i].legend[0], 'right')
    return plots

# 1. Define the original super-gaussian
mu = 0
sigma = 1.2
n = 3.4
x_original = np.linspace(degrees[0], degrees[1], number_points)
y_original = np.exp(-abs(((x-mu)/sigma))**n)

# 2. Create spline interpolation of average points
int_points_normalized = np.divide(int_points_average, np.max(int_points_average))
interp_plots= spline_interpolation(int_axis_average, int_points_normalized, x_original, width=500, height = 300)
grid_interp = gridplot(children = interp_plots, ncols = 3, merge_tools=False)
show(grid_interp)


# -

dataset=[[0, 5, 4, 0, 3, 6, 4, 5, 7, 11, 15, 27, 59, 1021, 24449, 35191, 35366, 26900, 4308, 78, 37, 20, 15, 7, 6, 6, -1, 2, 5, 2, 7, 5],
      [1, 3, 2, 6, 2, 4, 6, 8, 9, 13, 20, 38, 84, 9003, 30772, 35925, 32967, 17945, 143, 56, 27, 16, 12, 7, 6, 8, 0, 1, 3, 7, 7, 2],
      [1, 3, 0, 2, 6, 0, 4, 8, 10, 16, 23, 47, 102, 14001, 32542, 35503, 31287, 12760, 110, 51, 28, 14, 12, 7, 5, 5, 1, 5, 5, 7, 4, 2],
      [1, 6, 3, 5, 4, 6, 4, 10, 13, 21, 34, 70, 2262, 26211, 34648, 34221, 25418, 2748, 77, 44, 22, 12, 11, 7, 6, 6, 1, 4, 5, 5, 6, 4],
      [-1, 3, 2, 4, 4, 4, 4, 4, 6, 11, 14, 22, 42, 82, 9257, 31055, 35591, 32848, 17549, 142, 57, 30, 22, 7, 6, 7, 4, 3, 3, 3, 3, 4],
      [-2, 2, 4, 3, 3, 0, 6, 5, 4, 7, 10, 19, 30, 46, 98, 8420, 30114, 34256, 31826, 18618, 164, 69, 37, 20, 10, 13, 6, 4, 5, 9, 7, 3]
      ]
x = np.linspace(0, 31, 32)
Aq = [1.65, 1.70, 1.73, 1.69, 1.70, 1.74]
I = [1274, 1272, 1267, 1260, 1269, 1239]
M = [0.09, -0.343, -0.534, -1.013, 0.657, 1.704]
print(x)

plots = []
for i, data in enumerate(dataset):
    TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
    p = figure(title = f"Aq: {Aq[i]}; I: {I[i]}; M: {M[i]}", x_axis_label = "Sample #", y_axis_label = "Intensity", tooltips = TOOLTIPS, 
               width = 400, height = 300)
    p.line(x, data, line_width = 4, alpha = 0.7, color="#95D190")
    p.circle(x, data, size = 8)
    plots.append(p)
    plot_format(p, "Sample #", "Intensity", "top_left", "8pt", "8pt", "8pt")
plot_grid = gridplot(children=plots, ncols = 3)
show(plot_grid)


