# ---
# title: "Interleaving algorithm"
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
#| echo: false
from bokeh.plotting import figure, show
from bokeh.models import Range1d, Span
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
import bokeh.palettes
import numpy as np
import warnings
warnings.filterwarnings('ignore')
output_notebook()
palette = ["#f3c623", "#e84c3d", "#3d97e8", "#3dc8e8", "#6dc8e8", "#8e7fe5", "#f54c4f", "#4c4ff5", "#4cf5b7", "#f5b74c", "#b74cf5", "#4cf54c", "#f54c9c", "#9c4cf5", "#4cf59c", "#f59c4c", "#9c4c8f", "#4c8f9c", "#8f4c9c", "#9c8f4c"]

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
    plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"

    # Title format
    plot.title.text_font_size = titlesize
    plot.title.text_font_style = "bold"

    # Dark theme
    plot.background_fill_color = "#282B30"
    plot.border_fill_color = "#282B30"
    plot.xgrid.grid_line_color = '#606773'
    plot.ygrid.grid_line_color = '#606773'
    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    
   
    return plot


# -

# # <span style="color:#A6DDFF">Problem statement - interleaving algorithm</span>

# The sampled optical field on the line detector was previously simulated in notebook: [Optical field simulation](a_Optical_Simulation.ipynb).
#
# Some of the observations were:
#
# 1. There are 32 sampling points as the line detector has the same number of pixels. This was simulated using a window integration over a defined optical field defined by: $y = e^{-((x-\mu)/\sigma)^n}$. 
#
# 2. The parameters $\mu$ and $\sigma$ determine the lateral shift and widht of the gaussian function respectively. These parameters are related to the incoming light angle from the sample and the material roughness.
#
# 3. The $\mu$ parameter was swept to simulate different incoming light angles. With this data, a histogram was then reconstructed and its standard deviation value was calculated.  
#
# 4. An oscillating pattern was observed in the histogram's standard deviation (Aq parameter). This is not ideal as this indicated a roughness change, however only the angle of the incoming light was being swept.
#
# 5. Hence, a new parameter that is constant as a function of $\mu$ has to be found. 
#
# 6. It is proposed to implement an interleaving algorithm to increase the density of sampling points as shown in @fig-1
#

#
# ::: {#fig-1}
#
# ![](images/b/Fig_1.png)
#
# a. Generate gaussians with different $\mu$ values. b. Calculate a window integration and re-center sampling points. c. Interleave points to increase sampling density. d. Interpolate to get the underlying optical field.
# :::

# ## Generate gaussian values with different $\mu$ values and window integration

# The first step is to generate different optical values with varying $\mu values$. Mathematically, this represents a lateral shifht of the gaussian function. In practice, this is equivalent to sweeping the income angle of the light going into the line detector. The code generate gaussian values i

# +
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
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-abs(((x-mu)/sigma))**n)
    
    # 2. Plot 
    if plot:
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = width, height = height)
        vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
        p.add_layout(vline)
        p.line(x[::20], y[::20], line_width=4, alpha = 0.9, line_color = "#9DD9C5", legend_label = "Optical field")
        return p, x, y
    else:
        return x, y
    

def window_integration(number_windows, window_size, gap, x, y, mu, p=None, window_bool = False, plot_grid = False, interleave_bool = False):
    """
    Performs a window integration

    Parameters
    ----------
    number_windows (int): Number of integration windows
    window_size (int): Number of data points in the window
    gap(int): Number of gap points
    x(np): linspace for the gaussian plot
    y(np): gaussian values
    mu(float): Displacement mu
    p(bokeh plot)
    Returns
    -------
    p (bokeh plot): Plot of the integration
    integration_axis (np): window integration axis
    integration_points (np): Integrated points
    central_points(list): central point of each integration window
    """
    integration_points = []
    integration_axis = []
    central_points = []
    for i in range(number_windows):
        # 1. Get the data inside the window and substract the gap points
        a = i*window_size
        b = i*window_size + window_size
        x_temp = x[a:b-gap:1]
        y_temp = y[a:b-gap:1]

        # 2. Perform integration and append integrated value
        integration = np.trapz(y_temp, x_temp, dx = x[1] - x[0])
        integration_points.append(integration)

        # 3. Calculate the central and shifted point of the window
        central_point = x_temp[len(x_temp)//2]
        shifted_point = central_point - mu
        central_points.append(central_point)
        integration_axis.append(shifted_point)
        
        # 4. Plot the shifted sampled points by each mu
        if interleave_bool:
            interleaved_plot.circle(shifted_point, integration, size = 8, color=new_colors[i])
        
        # 5. Plot the integration window
        if window_bool:
            left_edge = x_temp[0]
            right_edge = x_temp[-1]
            interleaved_plot.rect(x=(left_edge + right_edge)/2, y=0.18, width=right_edge-left_edge, height=0.3, 
                                fill_alpha=0.001, fill_color='#C5E0B4', color='#C5E0B4')
            interleaved_plot.rect(x=(right_edge + x[b-1])/2, y=0.18, width=x[b-1]-right_edge, height=0.3, 
                                fill_alpha=0.005, fill_color='#F16C08', color = '#F16C08')
    
    # 6. Generate plot grid
    if plot_grid:
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 1)
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0')
        p = plot_format(p, "Degrees", "Intensity", "top_left", "10pt", "10pt", "8pt")
        p.x_range = Range1d(-16, 16)
        p.y_range = Range1d(-0.5, 2)
        p.xaxis.ticker.desired_num_ticks = 10
        
    integration_axis = np.array(integration_axis)
    integration_points = np.array(integration_points)
    return p, integration_axis, integration_points, central_points



# +
#| column: screen
# 1. Define input parameters for the optical field
mu_np = np.linspace(0, 1, 11)
std_np = np.linspace(1.2, 1.3, 1)
n = 3.4
number_points = 10000
degrees = [-15, 15]
number_windows = 32
window_size = number_points//number_windows
gap = 100
plots_gaussian = []

# 2. Sweep mu parameter and do windown integration
for i in range(len(mu_np)):
    for j in range(len(std_np)):
        # b1. Perform window integration
        title = f"mu: {mu_np[i]:.1f}, std: {std_np[j]:.3f}"
        p, x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, True, title, 320, 260)
        p, int_axis, int_points, central_points = window_integration(number_windows, window_size, gap, x, y, mu_np[i], p, False, True, False)
        plots_gaussian.append(p)

grid_gaussian = gridplot(children = plots_gaussian, ncols = 6, merge_tools=False)
show(grid_gaussian)
# -

# \
# The output plot shows the generated optical fields as a function of varying $\mu$. Notice how the optical field (green curve) is laterally shifted as a function of $\mu$. A window integration is then calculated obtaining 32 sampling points (pink points), which are then reshifted with respect to the zero degree position. These are the steps suggested in @fig-1 a,b.

# ## Interleaving algorithm

# As it has been previously mentioned, one of the data processing limitations is the low number of sampling points per optical field (32 sampling points). A proposed solution for this is an interleaving algorithm (fig-1 c).
#
#  This consists of increasing the density of sampling points by sweeping the $\mu$ parameter, calculating the window integration and then re-shifting back with respect to 0 degrees. The sampling points will be shifted according to the $\mu$ value, hence obtaining more sampling points. The implemeted code for this algorithm is:

# +
# a. Initiate interleaved plot and arrays
TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
interleaved_plot = figure(title='Interleaved points', x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = 680, height = 550, x_range=Range1d(-5, 5), y_range=Range1d(-0.2, 1.0))
int_axis_interleaved = []
int_points_interleaved = []

# b. Generate color list
color_multiplier = len(bokeh.palettes.Turbo256)//32
colors = [bokeh.palettes.Turbo256[color_multiplier*i] for i in range(32)]
new_colors = []
for i in range(len(colors) // 2):
        # new_colors.append(colors[i])
        # new_colors.append(colors[len(colors) - i - 1])
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')

# c. Define input parameters for the optical field
mu_np = np.linspace(0, 0.9, 10)
std_np = np.linspace(1.2, 1.3, 1)
n = 3.4
number_points = 9984
degrees = [-16, 16]
number_windows = 32
window_size = number_points//number_windows
gap = 20
plots_gaussian = []

# d. Sweep mu parameter and do windown integration
for i in range(len(mu_np)):
    for j in range(len(std_np)):
        # Perform window integration
        title = f"mu: {mu_np[i]:.1f}, std: {std_np[j]:.3f}"
        p, x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, True, title, 320, 260)
        p, int_axis, int_points, central_points = window_integration(number_windows, window_size, gap, x, y, mu_np[i], p, True, False, True)
        plots_gaussian.append(p)

    # Concatenate sampled points
    int_axis_interleaved.extend(int_axis)
    int_points_interleaved.extend(int_points)

interleaved_plot = plot_format(interleaved_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "8pt")
interleaved_plot.circle(central_points, 0.34, size = 8, alpha = 0.7, color = 'green')
interleaved_plot.xaxis.ticker.desired_num_ticks = 20
interleaved_plot.yaxis.ticker.desired_num_ticks = 10
show(interleaved_plot)
# -

# The previous plot consists of:
#
# 1. The rectangles represent the integration window and the gap points.
#
# 2. The green point on top of the rectangles represent the central point of the integration window.
#
# 3. The interleaved points (purple, blue, green), represent the shifted integrated points. For example, one block of purple points are the integrated points within window_n1 shifted from $\mu = 0$ to $\mu = 1.0$, the consecutive blue points are the integrated points within window_n2 and so on.
#
# 4. As expected the interleaving algorithm increases the sampling point density at different $\mu$ positions.

# ## Interpolation and correlation coefficient 

# +
# a. Initiate interleaved plot and arrays
TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
interleaved_plot_b = figure(title='Interleaved points - interpolation ', x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = 680, height = 550, x_range=Range1d(-5, 5), y_range=Range1d(-0.2, 1.5))
int_axis_interleaved_b = []
int_points_interleaved_b = []

# b. Generate color list
color_multiplier = len(bokeh.palettes.Turbo256)//32
colors = [bokeh.palettes.Turbo256[color_multiplier*i] for i in range(32)]
new_colors = []
for i in range(len(colors) // 2):
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')

# c. Define input parameters for the optical field
mu_np = np.linspace(0, 0.9, 10)
std_np = np.linspace(1.2, 1.3, 1)
sigma = std_np[0]
n = 3.4
number_points = 9984
degrees = [-16, 16]
number_windows = 32
window_size = number_points//number_windows
gap = 20


# d. Sweep mu parameter and do windown integration
for i in range(len(mu_np)):
    for j in range(len(std_np)):
        # Perform window integration
        title = f"mu: {mu_np[i]:.1f}, std: {std_np[j]:.3f}"
        x, y = plot_equation(mu_np[i], std_np[j], n, number_points, degrees, False, title, 320, 260)
        p, int_axis, int_points, central_points = window_integration(number_windows, window_size, gap, x, y, mu_np[i])
        # plots_gaussian.append(p)

    # Concatenate sampled points
    int_axis_interleaved_b.extend(int_axis)
    int_points_interleaved_b.extend(int_points)

# e. Interleave sampled points obtained from window integration and normalize the values
int_axis_interleaved_b, int_points_interleaved_b = zip(*sorted(zip(int_axis_interleaved, int_points_interleaved)))
int_points_interleaved_normalized_b = np.divide(int_points_interleaved_b, np.max(int_points_interleaved_b))

# # f. Calculate optical field with interleaved x axis and calculate correlation factor
mu = 0
y_original = np.exp(-abs(((np.array(int_axis_interleaved_b)-mu)/sigma))**n)
corr_coef = np.corrcoef(y_original, int_points_interleaved_normalized_b)[0,1]
interleaved_plot_b.title = f"Interleaved points and optical field; Corr. Coefficient: {corr_coef:.4f}"

# # f. Plot interleaved data
interleaved_plot_b.line(int_axis_interleaved_b, int_points_interleaved_normalized_b, line_width = 7, color='#4B9AFF',legend_label = 'Interleaved line')
interleaved_plot_b.circle(int_axis_interleaved_b, int_points_interleaved_normalized_b, size = 8, alpha = 1, legend_label = 'Interleaved points', color = '#4B9AFF')

# # g. Plot optical field
interleaved_plot_b.line(int_axis_interleaved_b, y_original, line_width = 7, alpha = 0.9, legend_label = 'Optical field line', color = '#9DD9C5')
interleaved_plot_b.circle(int_axis_interleaved_b, y_original, size = 8, legend_label = 'Optical field points', color = '#9DD9C5')


# # h. Format plot
interleaved_plot_b.xaxis.ticker.desired_num_ticks = 20
interleaved_plot_b.yaxis.ticker.desired_num_ticks = 10
interleaved_plot_b = plot_format(interleaved_plot_b, "Degrees", "Intensity", "top_left", "11pt", "13pt", "11pt")
show(interleaved_plot_b)
# -

# - Integrate the known optical field (green points)
# - So every degree I get 10 data points Convolve it with a 10 1's (smoothing window)
# - turn green line into blue line through convolution
#
#
# PT_wafer2 -  we know there is a clear roughness in some spots
# Take a narrow one and see if I can get angle/tilt value
#
# What does it take to make a best fit? this is the center/amplitudes are different
# That will be a fit angle
#
# Out all of those which ones are broader?
#
#
#
