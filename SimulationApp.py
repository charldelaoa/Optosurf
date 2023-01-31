import streamlit as st
import bokeh
from bokeh.plotting import figure, curdoc
from bokeh.models import Rect, LinearColorMapper, SingleIntervalTicker, LinearAxis, Grid
from bokeh.layouts import gridplot
import numpy as np
import altair as alt
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, BSpline
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
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"
    # plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.5

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
    plot.title.text_font_size = "15pt"

    plot.legend.click_policy="hide"
    return plot


# Create a function to plot the equation
def plot_equation(mu, sigma, n, number_points, degrees, plot, title="Optical field sampling", width = 600, height = 450):
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
    ticker = SingleIntervalTicker(interval=2.5, num_minor_ticks=10)
    xaxis = LinearAxis(ticker = ticker)
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-abs(((x-mu)/sigma))**n)
    
    # 2. Plot 
    if plot:
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = width, height = height)
        p.line(x, y, line_width=4, alpha = 0.5, legend_label="Optical field", color='#C5E0B4')
        p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        p = plot_format(p, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
        return p, x, y
    else:
        return x, y


# Create a function to do the window integration
def window_integration(number_windows, window_size, x, y, p=None):
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
            p.circle(x_temp[::15], y_temp[::15], size = 4, alpha = 1, legend_label="Optical field with gaps")
            count += 1
    if p is not None:
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0', legend_label="Integrated/sampling points")
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 0.8)
        p.legend.location = "top_left"
    return p, integration_axis, integration_points


# Create a function to do histogram reconstruction
def histogram_reconstruction(int_points, hist_bool):
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
        hist_plot = alt.Chart(pd.DataFrame({'x': hist_2d})).mark_bar(opacity=0.7, color='#2D908C').encode(
        alt.X('x:Q', bin=alt.Bin(maxbins=20)),
        y='count()', 
        tooltip=['x','count()'])
        title = f'Standard deviation: {stddev}'
        hist_plot = hist_plot.properties(title=title)
        return hist_plot, np.std(stddev)

    else:
        return stddev


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
                            ("Pchip", PchipInterpolator, '#EEDA89')]
                            #("Akima", Akima1DInterpolator, '#FEEED9')
    plots = [figure(width = 600, height = 400, tooltips = TOOLTIPS) for i in range(len(interpolation_methods))]
    plots_hist = [figure(width = 600, height = 400, tooltips = TOOLTIPS) for i in range(len(interpolation_methods))]
    
    for i, (method, interp_func, color) in enumerate(interpolation_methods):
        # a. Interpolate data  
        interp = interp_func(x, y)
        interp_points = interp(new_axis)
        
        # b. Calculate FWHM 
        fwhm = calculate_fwhm(new_axis, interp_points)
        # hist_plot, std_dev = histogram_reconstruction(interp_points, True)
        

        # b. Plot the data
        plots[i].line(new_axis, interp_points, line_width=6, color= color)
        plots[i].circle(x, y, size=7, color='#FAA0A0')
        plots[i].title = f"Method: {method}; FWHM: {fwhm}"
        plots[i] = plot_format(plots[i], "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
        plots_hist[i] = hist_plot
    return plots, plots_hist
    

def standard_matrix(mu_np, std_np, gaussian_grid_boolean):
    """
    Calculates the standard deviation of multiple histograms 

    Parameters
    ----------
    mu_np(np): range to vary mu parameter of gaussian function
    std_np(np): range of standard deviation to vary
    Returns
    -------
    std_grid(np): standard deviation matrix
    """
    X, Y = np.meshgrid(mu_np, std_np)
    std_grid = np.empty_like(X)
    plots_gaussian = []
   
    for i in range(len(mu_np)):
        for j in range(len(std_np)):
            # Generate x and y Gaussian data points 
            if gaussian_grid_boolean:
                title = f"mu: {mu_np[i]:.3f}, std: {std_np[j]:.3f}"
                plot, x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, True, title, 250, 250)
                plot.title.text_font_size = "10pt"
                plot, int_axis, int_points = window_integration(number_windows, window_size, x, y, plot)
                plots_gaussian.append(plot)
            else:
                # st.write(f"mu: {mu_np[i]}, std: {std_np[j]}")
                x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, False)
                p, int_axis, int_points = window_integration(number_windows, window_size, x, y)
            std_grid[j, i]  = histogram_reconstruction(int_points, False)

    source = pd.DataFrame({'mu': X.ravel(),
                        'std': Y.ravel(),
                        'value': std_grid.ravel()})

    return std_grid, source, plots_gaussian


# %% 1. Define the default values for the slider variables
st.title("Optical field definition as: $y = e^{-((x-\mu)/\sigma)^n}$")
st.markdown("## Optical field definition and window integration")
# a. Streamlit sliders -Gaussian parameters
st.sidebar.title("Gaussian parameters")
expander_g = st.sidebar.expander("Gaussian parameters", expanded = True)
with expander_g:
    mu = st.slider("Mean", -15.0, 15.0, 0.0, 0.1)
    sigma = st.slider("Standard Deviation", 0.1, 5.0, 1.3, 0.1)
    n = st.slider("Order", 0.0, 10.0, 3.5, 0.5)
    number_points = st.slider("Number of points", 0, 100000, 50000, 500)
    degrees = st.slider("Select degrees range", -30.0, 30.0, (-15.0, 15.0))
            
# b. Integration parameters
st.sidebar.title("Integration parameters")
expander_i = st.sidebar.expander("Integration parameters", expanded = True)
with expander_i:
    number_windows = st.slider("Number of windows", 1, 100, 32, 1)
    gap = st.slider("Number of gap points", 0, 1000, 100, 1)
    window_size = number_points//number_windows
    st.write('Window size: ', window_size)

# c. Standard deviation matrix parameters
st.sidebar.title("Standard deviation matrix parameters")
matrix_bool = st.sidebar.checkbox("Calculate standard deviation matrix", True)
if matrix_bool:
    expander_r = st.sidebar.expander("Standard deviation parameters", expanded = True)
    with expander_r:
        mu_range = st.slider("Select the median (mu) range", -15.0, 15.0, (-5.0, 5.0))
        mu_step = st.number_input("Input the mu step", 0.0, 10.0, 0.1)
        std_range = [sigma, sigma]
        std_step = st.number_input("Input the standard deviation step", 0.0, 5.0, 1.0)
        mu_points = (mu_range[1] - mu_range[0])//mu_step
        std_points = (std_range[1] - std_range[0])/std_step
        gaussian_grid_boolean = st.checkbox("Plot Gaussian grid", False)
        if not gaussian_grid_boolean:
            mu_np = np.linspace(mu_range[0], mu_range[1], int(mu_points))
            std_np = np.linspace(std_range[0], std_range[1], int(std_points)+1)
        else:
            mu_np = np.linspace(mu_range[0], mu_range[1], 6)
            std_np = np.linspace(std_range[0], std_range[1], 6)
        st.write('mu array size', len(mu_np))
        st.write('std array size', len(std_np))

# d. Spline interpolation parameters
st.sidebar.title("Spline interpolation parameters")
expander_s = st.sidebar.expander("Spline interpolation parameters", expanded = True)
with expander_s:
    degrees_s = st.slider("Select spline degrees range", -30.0, 30.0, (-15.0, 15.0))
    num_points_s = st.number_input("Number of points spline interpolation", 0, 100000, 1000)


# %% 2. Plot optical field equation
# x(np): linspace for the gaussian plot
# y(np): gaussian values
p, x, y = plot_equation(mu, sigma, n, number_points, degrees, True)


# %% 3. Compute window integration
p, int_axis, int_points = window_integration(number_windows, window_size, x, y, p)


# %% 4. Make a 2D histogram reconstruction
hist_plot, std_dev = histogram_reconstruction(int_points, True)


# %% 5. Create standard deviation matrix
if matrix_bool:
    std_grid, source, plots_gaussian = standard_matrix(mu_np, std_np, gaussian_grid_boolean)

    if gaussian_grid_boolean:
        grid_gaussian = gridplot(children = plots_gaussian, ncols = len(std_np), merge_tools=False)
        st.bokeh_chart(grid_gaussian)


# %% 6. Plot optical field, histogram and standard deviation matrix
# a. Window integration and histogram reconstruction
col1, col2 = st.columns(2)
with col1:
    st.bokeh_chart(p)
    st.markdown("- The optical field is defined according to the equation $y = e^{-((x-\mu)/\sigma)^n}$ (green curve)")
    st.markdown(f"- Sampling gaps are simulated (blue curve)")
    st.markdown(f"- The pixel sampling is simulated by a window integration using {number_windows} windows (pink points)")

with col2:
    hist_plot.width = 500
    hist_plot.height = 450
    st.altair_chart(hist_plot, use_container_width=False)
    st.markdown(f"- A histogram is reconstructed from the sampled 32 points")
    st.markdown(f"- The standard deviation is then calculated")


# b. Plot standard deviation matrix
if matrix_bool:
    st.title("Standard deviation matrix")
    st.markdown("A standard deviation matrix is calculated by sweeping the $\mu$ parameter of the optical field at a constant standard deviation $\sigma$")
    st.markdown(f"The mu parameter is related to the angle of the incoming light and is swept from {mu_range[0]} to {mu_range[1]} with {mu_step} step")
    st.markdown(f"Notice the variation in the calculated standard deviation from every histogram as plotted in the matrix, this means that this value is not a constant function of $\mu$")

    axis_range = np.arange(-5, 5.5, 0.5)
    intensity_plot = alt.Chart(source).mark_rect().encode(
        alt.X('mu:O', axis=alt.Axis(values=axis_range, format=".1f")),
        y='std:O',
        color='value:Q',
        tooltip='value')
    intensity_plot = intensity_plot.properties(width = 800, height = 300, title=f"Sweep in mu parameter at a constant std deviation value of {std_range[0]}")
    st.altair_chart(intensity_plot, use_container_width=True)


# %% 7. Spline interpolation
new_axis = np.linspace(degrees_s[0],degrees_s[1],num_points_s)
interp_plots, hist_plots = spline_interpolation(int_axis, int_points, new_axis)

# b. Spline interpolation plot
st.title(f"Spline interpolation with {num_points_s} points")
grid_interp = gridplot(children = interp_plots, ncols = 2, merge_tools=False)
st.bokeh_chart(grid_interp)



# spline_plot = figure (title=f"{name} interpolation", x_axis_label="Sampling point", y_axis_label="Intensity",
#                         tooltips=tooltips, width = width, height = height)
#         spline_plot.circle(x, y, size = 7, color = '#FAA0A0', legend_label = "Integration points")
        
#         if method == np.interp:
#             points = method(new_axis, x, y)
#         else:
#             interp = method(x, y)
#             points = interp(new_axis)
            
#         spline_plot.line(new_axis, points, line_width=6, alpha = 0.5, 
#                          legend_label = f"{name} interpolation", line_dash=line_dash)
#         spline_plot = plot_format(spline_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
#         plots.append(spline_plot)
        
#     return plots

# 1. Try to interpolate with less points and reconstruct histogram calc. std. deviation plot matrix
# 2. Change angle axis from -5 to 5 and calculate parameters (HW solution)