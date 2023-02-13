import streamlit as st
from bokeh.plotting import figure
from bokeh.models import Range1d, Span
from bokeh.layouts import gridplot
import bokeh.palettes
import numpy as np
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, BSpline

st.set_page_config(page_title="Interleave", layout="wide")
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
            width = width, height = height)
        # x_range=Range1d(-5, 5), y_range=Range1d(-0.5, 1.2)
        vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
        p.add_layout(vline)
        p.line(x[::20], y[::20], line_width=4, alpha = 0.5, line_color = "#4B9AFF", legend_label = "Optical field")
        # p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        # p.add_layout(xaxis, 'below')
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
        p.xaxis.ticker.desired_num_ticks = 10
    
    # 6. Generate plot grid
    if plot_grid:
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 1)
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0', legend_label = 'Sampled Points')
        p = plot_format(p, "Degrees", "Intensity", "top_left", "10pt", "10pt", "8pt")
        p.x_range = Range1d(-16, 16)
        p.y_range = Range1d(-0.5, 2)
    integration_axis = np.array(integration_axis)
    integration_points = np.array(integration_points)
    return p, integration_axis, integration_points, central_points


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
    number_points = st.slider("Number of points", 0, 100000, 10000, 500)
    degrees = st.slider("Select degrees range", -30.0, 30.0, (-16.0, 16.0))

# b. Integration parameters
st.sidebar.title("Integration parameters")
expander_i = st.sidebar.expander("Integration parameters", expanded = False)
with expander_i:
    st.markdown("These parameters are the number of integration windows and gap points")
    number_windows = st.slider("Number of windows", 1, 100, 32, 1)
    gap = st.slider("Number of gap points", 0, 1000, 100, 1)
    window_size = number_points//number_windows
    st.write('Window size: ', window_size)

# c. Interleaved plot parameters
st.sidebar.title("Interleaved plot parameters")
x_plot = st.sidebar.slider("Plot xscale", -20.0, 20.0, (-5.0, 5.0))
y_plot = st.sidebar.slider("Plot yscale", -20.0, 20.0, (-0.3, 1.5))
interleave_bool = st.sidebar.checkbox('Plot interleaved points', False)
window_bool = st.sidebar.checkbox("Plot integration windows", False)
plot_grid = st.sidebar.checkbox('Plot gaussian grid', False)
# %% 2. Define starting plot grid and perform window integration
# a. Initiate interleaved plot and arrays
TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
interleaved_plot = figure(title='Interleaved points', x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = 800, height = 550, x_range=Range1d(x_plot[0], x_plot[1]), y_range=Range1d(y_plot[0], y_plot[1]))

mu_np_a = np.array(mu_options)
plots_gaussian = []
int_axis_interleaved = []
int_points_interleaved = []

# b. Generate color list
color_multiplier = len(bokeh.palettes.Turbo256)//32
colors = [bokeh.palettes.Turbo256[color_multiplier*i] for i in range(32)]
new_colors = []
for i in range(len(colors) // 2):
        new_colors.append(colors[i])
        new_colors.append(colors[len(colors) - i - 1])

# c. Sweep the mu parameter, perform window integration and concatenate sampled points
for i in range(len(mu_np_a)):
    for j in range(len(std_np)):
        # b1. Perform window integration
        title = f"mu: {mu_np_a[i]:.1f}, std: {std_np[j]:.3f}"
        p, x, y = plot_equation(mu_np_a[i], std_np[j], n, number_points, degrees, True, title, 320, 260)
        p, int_axis, int_points, central_points = window_integration(number_windows, window_size, gap, x, y, mu_np_a[i], p, window_bool, plot_grid, interleave_bool)
        plots_gaussian.append(p)
      
        # b2. Concatenate sampled points
        int_axis_interleaved.extend(int_axis)
        int_points_interleaved.extend(int_points)

# d. Interleave sampled points obtained from window integration and normalize the values
int_axis_interleaved, int_points_interleaved = zip(*sorted(zip(int_axis_interleaved, int_points_interleaved)))
int_points_interleaved_normalized = np.divide(int_points_interleaved, np.max(int_points_interleaved))

# e. Calculate optical field with interleaved x axis and calculate correlation factor
mu = 0
y_original = np.exp(-abs(((np.array(int_axis_interleaved)-mu)/sigma))**n)
corr_coef = np.corrcoef(y_original, int_points_interleaved_normalized)[0,1]
interleaved_plot.title = f"Interleaved points and optical field; Corr. Coefficient: {corr_coef:.4f}"

# f. Plot interleaved data
interleaved_plot.line(int_axis_interleaved, int_points_interleaved_normalized, line_width = 7, color='#4B9AFF',legend_label = 'Interleaved line')
interleaved_plot.circle(int_axis_interleaved, int_points_interleaved_normalized, size = 8, alpha = 1, legend_label = 'Interleaved points', color = '#4B9AFF')
y_points = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 1.2, 1.2, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# g. Plot optical field
interleaved_plot.line(int_axis_interleaved, y_original, line_width = 7, alpha = 0.9, legend_label = 'Optical field line', color = '#EEDA89')
interleaved_plot.circle(int_axis_interleaved, y_original, size = 8, legend_label = 'Optical field points', color = '#E5E863')
if window_bool:
    interleaved_plot.circle(central_points, 0.34, size = 8, alpha = 0.7, color = 'green')

# h. Format plot
interleaved_plot.xaxis.ticker.desired_num_ticks = 20
interleaved_plot.yaxis.ticker.desired_num_ticks = 10
interleaved_plot = plot_format(interleaved_plot, "Degrees", "Intensity", "top_left", "13pt", "15pt", "13pt")
st.bokeh_chart(interleaved_plot, use_container_width=True)

# i. Plot gaussian grid
if plot_grid:
    grid_gaussian = gridplot(children = plots_gaussian, ncols = 4, merge_tools=False)
    st.bokeh_chart(grid_gaussian)



# Now we have 320 points after interleaving - spline interplation, e.g.
# generated another optical field with mu = 0.4173 integrate that will give 32 points
# then can I use the 320 points to find out what mu value



