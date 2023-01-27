import streamlit as st
import bokeh
from bokeh.plotting import figure, curdoc
from bokeh.models import Rect, LinearColorMapper, SingleIntervalTicker, LinearAxis, Grid
from bokeh.layouts import gridplot
import numpy as np
import altair as alt
import pandas as pd

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
    return plot




# Create a function to plot the equation
def plot_equation(mu, sigma, n, number_points, degrees, plot, title="Super-Gaussian", width = 700, height = 550):
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
    # y = np.exp(-((x-mu)/sigma)**n)
    y = np.exp(-abs(((x-mu)/sigma))**n)
    
    # 2. Plot 
    if plot:
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = width, height = height)
        p.line(x, y, line_width=4, alpha = 0.5)
        p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        p = plot_format(p, "Degrees", "Intensity", "bottom_left", "10pt", "10pt", "10pt")
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
            p.circle(x_temp[::15], y_temp[::15], size = 4, alpha = 1)
            count += 1
    if p is not None:
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0')
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 0.8)
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


# %% 1. Define the default values for the slider variables
st.title("Super-Gaussian Equation Plotter plots: $y = e^{-((x-\mu)/\sigma)^n}$")

# a. Streamlit sliders -Gaussian parameters
st.sidebar.title("Gaussian parameters")
expander_g = st.sidebar.expander("Gaussian parameters", expanded = True)
with expander_g:
    mu = st.slider("Mean", -15.0, 15.0, 0.0, 0.1)
    sigma = st.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
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

# c. Standard deviation parameters
st.sidebar.title("Standard deviation parameters")
matrix_bool = st.sidebar.checkbox("Calculate standard deviation matrix", False)

if matrix_bool:
    expander_r = st.sidebar.expander("Standard deviation parameters", expanded = True)
    with expander_r:
        mu_range = st.slider("Select the median (mu) range", -15.0, 15.0, (-5.0, 5.0))
        mu_step = st.number_input("Input the mu step", 0.0, 10.0, 0.1)
        std_range = st.slider("Select the standard deviation range", 0.0, 5.0, (1.5, 1.5))
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
# %% 2. Plot gaussian equation
# x(np): linspace for the gaussian plot
# y(np): gaussian values
p, x, y = plot_equation(mu, sigma, n, number_points, degrees, True)


# %% 3. Compute window integration
p, int_axis, int_points = window_integration(number_windows, window_size, x, y, p)


# %% 4. Make a 2D histogram reconstruction
# a. Histogram reconstruction
hist_plot, std_dev = histogram_reconstruction(int_points, True)


# %% 5. Plot column layout
col1, col2 = st.columns(2)
with col1:
    st.bokeh_chart(p)

with col2:
    hist_plot.width = 500
    hist_plot.height = 550

    st.altair_chart(hist_plot, use_container_width=True)


# %% 6. Standard deviation 2D plot

# a. Create a grid with the input mu and std_dev
if matrix_bool:
    X, Y = np.meshgrid(mu_np, std_np)
    std_grid = np.empty_like(X)
    st.write(std_grid.shape)
    plots_gaussian = []
    plots_histogram = []
    n
    number_points
    degrees
    # Iterates mu and standard deviation
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
            
    if gaussian_grid_boolean:
        grid_gaussian = gridplot(children = plots_gaussian, ncols = len(std_np), merge_tools=False)
        st.bokeh_chart(grid_gaussian)

    source = pd.DataFrame({'mu': X.ravel(),
                        'std': Y.ravel(),
                        'value': std_grid.ravel()})

    intensity_plot = alt.Chart(source).mark_rect().encode(
        x='mu:O',
        y='std:O',
        color='value:Q',
        tooltip='value')
    intensity_plot = intensity_plot.properties(width = 1300, height = 850)
    st.altair_chart(intensity_plot)
# for i in range(len(mu_np)):
#     for j in range(len(std_np)):
#         # Calculate Gaussian
#         x = np.linspace(degrees[0], degrees[1], number_points)
#         y = np.exp(-((x-mu_np[i])/std_np[j])**n)
#         integration_points = []
#         integration_axis = []
#         for z in range(number_windows):
#             a = z*window_size
#             b = z*window_size + window_size

#             x_temp = x[a:b-gap:1]
#             y_temp = y[a:b-gap:1]
#             integration = np.trapz(y_temp, x_temp, dx = x[1] - x[0])
#             integration_points.append(integration)

#             axis = x_temp[len(x_temp)//2]
#             integration_axis.append(axis)
        # st.write('integration points', integration_points)
        # st.write('integration axis', integration_axis)
        # integration_axis
        # std_grid[j,i] = i*j


        # st.write(f"i: {i}, j: {j}")
        # st.write(f"mu: {mu_np[i]}, std: {std_np[j]}")


    # X[i]









# normalized_y = np.multiply(int_points, 100000)
# hist_2d = np.array([])

# for i, int_point in enumerate(normalized_y):
#     round_int_point = round(float(int_point))
#     hist_2d = np.concatenate((hist_2d, np.array([i]*round_int_point)))

# # b. Plot histogram
# hist_plot = figure(title = "Histogram")
# hist, edges = np.histogram(hist_2d, bins = 20)

# hist_plot.quad(bottom=0, top=hist, left= edges[:-1], right=edges[1:], 
#                 fill_color='blue', line_color = 'white')

# st.write('Std deviation', np.std(hist_2d))

#unique_values, counts = np.unique(hist_2d, return_counts=True)
# Print the unique values and their occurrences
# for value, count in zip(unique_values, counts):
#     st.write(f'Value: {value}, Occurrences: {count}')

# hist_2d = []
# for i, int_point in enumerate(normalized_y):
#     hist_2d.extend([float(int_point)]*i)
# hist_2d


# Here's a code snippet from my Labview:

# c=[];
    # for indx2=1:32
    #     c=[c;indx2*ones(x),1)];
    # end
# stddev=std(c);

# where 'hout' is a 32 element vector with the Y values and 'c' becomes a 'histogram vector'. Y values went up to about 1500, so I didn't need to renormalise them. If you multiply your Y values (which go up to 1) by 1000, then you should be ok.







# 4. Plot for different mean values

# # a. Define np linear arrays for mean and standard deviation
# mean_points = st.sidebar.number_input("Number of points for mu value: ", 1, 50, 31)
# mean_np = np.linspace(-15, 15, mean_points)
# stand_dev_points = st.sidebar.number_input("Number of points for standard deviation value: ", 1, 101, 50)
# stand_dev_np = np.linspace(0.1, 5, stand_dev_points)

# # b. Create a meshgrid


# # d. Plot grid
# color_mapper = LinearColorMapper(palette="Viridis256", low=-75, high=75)
# grid = figure(title="product_grid")

# # Add the image to the figure
# hover = HoverTool(tooltips=[("index", "$index"),("(x,y)", "($x, $y)"), ("Intensity", "@image{0.00}")])
# grid.add_tools(hover)
# grid.image(image=[product_grid], x=-15, y=0, dw=30, dh=5, color_mapper=color_mapper)
# color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
#                      location=(0,0))
# grid.add_layout(color_bar, 'right')
# grid = plot_format(grid, "Mean", "Standard Deviation", "bottom_left", "10pt", "10pt", "10pt")
# st.bokeh_chart(grid)
# st.write(np.min(product_grid))

# %%
# In simulation i am generating 32 points
# make fit to pink data using gaussian function, will you get same parameters
# 1. From the pink data fit to gaussian and measure std deviation
# 2. Make a cubic spline interpolation of the pink data
# when we made sigma bigger there were more data points in pink data, the mu made no difference
# when there more data points 
# fewer data points it went bad, make more inter
# make interpolation to a new axis (0.1 degree),
# 
