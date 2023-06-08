import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bokeh.plotting import figure, show, save, output_file
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import Span
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from bokeh.models import ColumnDataSource
st.set_page_config(page_title="Base function optimization", layout="wide")


def plot_format(plot, xlabel, ylabel, location, size, titlesize, labelsize):
    # x axis format
    plot.xaxis.axis_label = xlabel
    plot.xaxis.axis_label_text_font_size = size
    plot.xaxis.axis_line_color = '#282B30'
    plot.xaxis.major_tick_line_color = '#DAE3F3'
    plot.xaxis.minor_tick_line_color = '#DAE3F3'
    plot.xaxis.major_label_text_font_size = size
    
    # y axis format
    plot.yaxis.axis_label = ylabel
    plot.yaxis.axis_label_text_font_size = size
    plot.yaxis.major_label_text_font_size = size
    plot.yaxis.axis_line_color = '#282B30'
    
    # Legend format
    plot.legend.location = location
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = labelsize
    plot.legend.border_line_width = 3
    plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.0
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"

    # Title format
    plot.title.text_font_size = titlesize
    plot.title.text_font_style = "bold"
    plot.outline_line_color = '#282B30'

    # Dark theme
    plot.background_fill_color = "#282B30"
    plot.border_fill_color = "#282B30"
    plot.xgrid.grid_line_color = '#606773'
    # plot.xgrid.minor_grid_line_color = '#606773' 
    # plot.xgrid.minor_grid_line_alpha = 0.4
    # plot.xgrid.minor_grid_line_dash = [2, 2] 
    plot.xaxis.minor_tick_line_color = '#606773'
    plot.yaxis.minor_tick_line_color = '#606773'
    plot.ygrid.grid_line_color = '#606773'
    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    return plot

new_colors = []
for i in range(42):
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')
        new_colors.append('#F2DDA4')
        new_colors.append('#C4A287')
        new_colors.append('#FFB5C2')
        new_colors.append('#CD533B')
iterations = {}
iteration_1 = {}
iterations_shifted = {}
iteration_1_shifted = {}
TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
# window_size = 0.05

def subplot3d(files, offaxis, onaxis, rows, cols, path, slice1, slice2):
    """
    Generates a 3D plot of the data

    Parameters
    ----------
    files (list): List of files to plot
    df_slice (dataframe): Dataframe with the slices to plot
    
    Returns
    -------
    subplot (plotly.graph_objects.Figure): Plotly figure with the 3D plot
    spots (dict): Dictionary with the data of the files
    """
    # a. Define on and off axis as well as subplots
    subplot = make_subplots(rows=rows, cols=cols, subplot_titles=files, 
                    specs=[[{'type': 'surface'}]*cols]*rows)
    x = np.linspace(0, 10, 100)
    z = np.linspace(0, 10, 100)
    y = np.linspace(-15.5, 15.5, 100)
    X, Z = np.meshgrid(x, z)
    Y, Z2 = np.meshgrid(y, z)
    count = 0
    spots = {}
    col = 1

    # b. Read file and create matrix
    for i, file in enumerate(files):
        file_path = path + file
        # c. Read the .csv file
        vals = np.genfromtxt(file_path, delimiter=',')
        # vals = vals[:, 1200:2800]
  
        # d. Normalize values
        normalized_vals = vals[:, 1:] 

        # e. 3D surface plot
        # surface = go.Surface(x=offaxis[1200:2800], y=onaxis, z=normalized_vals, colorscale='jet', 
        #            showscale=False)
        surface = go.Surface(x=offaxis, y=onaxis, z=normalized_vals, colorscale='jet', 
                   showscale=False)
        # add Slice plane
        slice_on = go.Surface(x=offaxis[slice1]*np.ones_like(Y), y=Y, z=3500*Z2, opacity=0.6, showscale=False, colorscale='Viridis')
        slice_on_2 = go.Surface(x=offaxis[slice2]*np.ones_like(Y), y=Y, z=3500*Z2, opacity=0.6, showscale=False, colorscale='Viridis')
        
        row = i // 3 + 1  # Compute the row index based on the file index
        col = i % 3 + 1   # Compute the column index based on the file index
        subplot.add_trace(surface, row=row, col=col)
        subplot.add_trace(slice_on, row=row, col=col)
        subplot.add_trace(slice_on_2, row=row, col=col)

        # f. Create dictionary
        vals = np.genfromtxt(file_path, delimiter=',')
        spots[file] = vals
    return subplot, spots

# 1. Define files 
file_Ann_7 = 'Ann_7_70deg.csv' 
file_Ann_8b = 'Ann_8b_W2_70deg.csv'
file_Ann_8d = 'Ann_8d_C2_70deg.csv'
file_Ann_10 = 'Ann_10_70deg.csv'
file_PT = 'PT_SiRef_shiny_70.csv'

# 2. Slice/ displacement / window_size sliders
slice1 = st.sidebar.number_input('Slice 1', 0, 4096, 1974)
slice2 = st.sidebar.number_input('Slice 2', 0, 4096, 2002)
displacement = st.sidebar.number_input('Mechanical axis displacement', -10.0, 10.0, 0.0)
window_size = st.sidebar.number_input('Window size', 0.000, 1.0000, 0.05)
offaxis_angle = np.arange(-2048, 2048, 1)*140/4096 + displacement

# 3. Define axis and subplots
path = '/Users/carlosreyes/Desktop/Academics/postdoc/confirm/python/optosurf/Streamlit/data/f/reference/'
st.sidebar.markdown(f'{offaxis_angle[slice1]:.2f}, {offaxis_angle[slice2]:.2f}')
offaxis = np.arange(0, 4096, 1)
onaxis=np.arange(-15.5,16.5,1)
# subplot, spots = subplot3d([file_Ann_7, file_Ann_8b, file_Ann_8d, file_Ann_10, file_PT], 
#                            offaxis_angle, onaxis, 2, 3, path, slice1, slice2)
subplot, spots = subplot3d([file_Ann_7, file_Ann_8b, file_Ann_8d, file_Ann_10, file_PT], 
                           offaxis, onaxis, 2, 3, path, slice1, slice2)

# 4. Camera settings
st.sidebar.markdown('Camera')
x = st.sidebar.slider('x', -1.0, 1.0, 0.25)
y = st.sidebar.slider('y', -1.0, 1.0, 0.5)
z = st.sidebar.slider('z', -1.0, 2.0, 1.0)
subplot.update_scenes(camera_eye=dict(x=x, y=y, z=z))
subplot.update_layout(width=1200, height=900)

plot_datasets = st.checkbox('Plot datasets', 'True')
if plot_datasets:
    st.plotly_chart(subplot)


# %% 2. 1st minimization with original optosurf axis
def cost_function(params, y, x):
    """
    Generates a 3D plot of the data

    Parameters
    ----------
    params (list): List of parameters to fit
    y (array): Array with the data to fit
    
    Returns
    -------
    rmse (float): Root mean square error of the fit0.
    """
    x0, A0 = params
    x_new = x + x0
    y_modified = A0*pchip(x_new)

    mse = np.mean((y - y_modified) ** 2)
    rmse = np.sqrt(mse)
    return rmse

slices = np.arange(slice1, slice2+1, step=1)
method = 'Powell'

# 2.1 Get initial base function
smooth_df = pd.read_csv("data/f/smooth_df.csv")
x_base = smooth_df['xaxis']
y_base = smooth_df['yaxis']
pchip = PchipInterpolator(x_base, y_base)

# 2.2 Define mechanical axis and optosurf axis
onaxis = np.arange(-15.5,16.5,1)
offaxis = np.arange(-2048, 2048, 1)*140/4096 + displacement
files = [file_Ann_7, file_Ann_8b, file_Ann_8d, file_Ann_10, file_PT]
file_choice = st.sidebar.radio('Select file to be used as base function', files)

minimized_df_dict = {}
minimized_df = pd.DataFrame(columns=["angle", "x0", "difference", "slice", "amplitude", "rmse"])
dataset = spots[file_choice]
for j, slice in enumerate(slices):
    # 2.3 Get initial guess
    x0 = offaxis[slice]
    Abase = 1.0
    guess_no_back = [x0, Abase]

    # 2.4 Call minimization function
    y = dataset[:,slice]
    cost_fn = lambda p:cost_function(p, y, onaxis)
    result = minimize(cost_fn, guess_no_back, method=method,)
    optimized_parameters = list(result.x)
    x0_opt, A0_opt, = optimized_parameters
    x_new_opt = onaxis + (x0_opt)
    y_optimized = A0_opt*pchip(x_new_opt)
    
    # Calculate error
    mse = np.mean((y - y_optimized) ** 2)
    rmse = np.sqrt(mse)
    row = [offaxis[slice], x0_opt, offaxis[slice] - x0_opt, slice, A0_opt, rmse]
    minimized_df.loc[j] = row
    
minimized_df_dict[file_choice] = minimized_df.copy()    
# st.write(minimized_df_dict[file_choice])
# %% 3. Error calculation

p1 = figure(title=f'a. Rotation angle vs. x0', tooltips = TOOLTIPS)
p2 = figure(title='c. Rotation angle vs. RMSE', tooltips = TOOLTIPS, y_range=(0, 100))
p3 = figure(title='b. x0 vs. difference (mechanical angle - x0)', tooltips = TOOLTIPS)
p4 = figure(title='d. Rotation angle vs. amplitude', tooltips = TOOLTIPS)
p5 = figure(title='Shifted optosurf axis', width=1000, height=350, x_range=(-28,28), tooltips = TOOLTIPS)

shifted_axis_dict = {}
angles_interp_dict = {}
amplitudes_interp_dict = {}

angles = minimized_df_dict[file_choice]['angle'].values
x0 = minimized_df_dict[file_choice]['x0'].values
# 3.1 Rotation angle vs. x0
p1.line(x=angles, y=x0, line_width=2, color = new_colors[0], legend_label=file_choice)

# 3.2 Rotation angle vs RMSE
p2.line(x=minimized_df_dict[file_choice]['angle'], y=minimized_df_dict[file_choice]['rmse'], line_width=2, color = new_colors[0], legend_label = file_choice)

# 3.3 Polynomial fit of x0 vs difference
x0 = minimized_df_dict[file_choice]['x0'].values
angles = minimized_df_dict[file_choice]['angle'].values
difference = minimized_df_dict[file_choice]['difference'].values
poly = PolynomialFeatures(degree=1)
x0_poly = poly.fit_transform(x0.reshape(-1,1))
poly_model = LinearRegression().fit(X=x0_poly, y=difference)
xaxis = np.arange(-15.5, 15.5, 0.1)
xaxis_poly = poly.transform(xaxis.reshape(-1,1))
ypoly = poly_model.predict(xaxis_poly)
p3.line(x=x0, y=difference, line_width=2, color = new_colors[0], legend_label = file_choice)
p3.line(x=xaxis, y=ypoly, line_width=2, color = new_colors[4], dash='dashed')

# 3.4 Calculate shifted axis
onaxis_poly = poly.transform(onaxis.reshape(-1,1))
y_shifted = poly_model.predict(onaxis_poly)
p3.circle(onaxis, y_shifted, size = 6, fill_color = new_colors[0], color='blue')
shifted_axis = onaxis + y_shifted
shifted_axis_dict[file_choice] = shifted_axis

# 3.5 Rotation angle vs amplitude with interpolation
angles = minimized_df_dict[file_choice]['angle'].values
amplitudes = minimized_df_dict[file_choice]['amplitude'].values
p4.line(x=angles, y=amplitudes, line_width=1, color = new_colors[0], legend_label = file_choice)

# 6. Update shifted axis plot 
p5.circle(x=shifted_axis, y=np.zeros(len(onaxis))+0.5*0, line_width=2, color = new_colors[0], size = 7, legend_label=file_choice)

# 7. Plots format
p1.xaxis.ticker.desired_num_ticks = 10
p1 = plot_format(p1, "Rotation angle", "x0", "top_left", "10pt", "8pt", "8pt")
p2.xaxis.ticker.desired_num_ticks = 10
p2 = plot_format(p2, "Rotation angle", "RMSE", "top_left", "10pt", "11pt", "8pt")
p3.xaxis.ticker.desired_num_ticks = 10
p3 = plot_format(p3, "x0", "difference", "top_left", "10pt", "11pt", "8pt")
p4.xaxis.ticker.desired_num_ticks = 10
p4 = plot_format(p4, "Rotation angle", "amplitude", "top_right", "10pt", "11pt", "8pt")

p5.circle(x=onaxis, y=np.zeros(len(onaxis))-0.5, line_width=2, color = new_colors[0+1], size = 7, legend_label='Original axis')
p5 = plot_format(p5, "Angle", "", "top_left", "10pt", "11pt", "8pt")
p5.xaxis.ticker.desired_num_ticks = 20

grid = gridplot(children=[p1, p2, p3, p4], ncols=2, merge_tools=False, width = 500, height = 400)
st.bokeh_chart(grid)
st.bokeh_chart(p5)
st.write(minimized_df_dict[file_choice])
temp_df = minimized_df_dict[file_choice].copy()
temp_df = temp_df.loc[temp_df['x0'] < 0.0]
last_slice = temp_df['slice'].values[-1]
# st.write(temp_df)
st.write(last_slice)


# %% 4. Base function recreation
def average_base(base_function, window_size):
    x_filtered = base_function['xaxis'].values
    y_filtered = base_function['yaxis'].values
    x_averaged = []
    y_averaged = []
    for i in np.arange(np.min(x_filtered), np.max(x_filtered), window_size):
        # get the indices of the points within the current window
        indices = np.where((x_filtered >= i) & (x_filtered < i + window_size))[0]
        if len(indices) == 0:
            continue
        # calculate the average x and y values for the points in the current window
        x_avg = np.mean(x_filtered[indices])
        y_avg = np.mean(y_filtered[indices])
        x_averaged.append(x_avg)
        y_averaged.append(y_avg)
    return x_averaged, y_averaged

slices_base = np.arange(slice1, slice2+1, step=1)
base_function_dict = {}
base_function_plots = []
TOOLTIPS2 = [
    ("x", "@xaxis"),
    ("y", "@yaxis"),
    ("slice", "@slice")
]

base_functions_plot = figure(title='Base functions', width=850, height=650, tooltips = TOOLTIPS2)

# 4.1 Get dataset, minimized df and shifted axis per file
vals_temp = spots[file_choice]
minimized_df_temp = minimized_df_dict[file_choice].set_index('slice')
shifted_axis_temp = shifted_axis_dict[file_choice]
x_base_array = []
y_base_array = []
slice_array = []

# 4.2 Iterate through slices
for k, slice in enumerate(slices_base):
    # 4.3 Get individual slice with 32 points and its corresponding correction factors
    y_temp = vals_temp[:,slice]
    angle = minimized_df_temp.loc[slice, 'angle']
    
    # 4.4 Reshift back to center
    new_shifted_axis = shifted_axis + angle
    x_base_array.extend(list(new_shifted_axis))
    y_base_array.extend(list(y_temp))
    slice_array.extend(np.ones(len(new_shifted_axis))*slice)

# 4.4 Create df with base function
base_function_df = pd.DataFrame({'xaxis': x_base_array, 'yaxis': y_base_array, 'slice': slice_array})
base_function_df = base_function_df.sort_values(by='xaxis')

# 4.5 Merge base function with minimized df
base_function_merged = base_function_df.merge(minimized_df_temp, on=['slice', 'slice'])
source = ColumnDataSource(base_function_merged)

# 4.6 Calculate average base function
x_averaged, y_averaged = average_base(base_function_merged, window_size)
pchip_base = PchipInterpolator(x_averaged, y_averaged)
x_base_interp = np.linspace(-15, 15, 1000)
y_base_interp = pchip_base(x_base_interp)

# 4.7 Plot base function
base_functions_0 = figure(title=f'Base function ({file_choice})' , width=850, height=650, tooltips = TOOLTIPS2,)
base_functions_0.circle(x='xaxis', y='yaxis', source = source, line_width=2, size = 6, legend_label='Base function')
base_functions_0.circle(x=x_averaged, y=y_averaged, line_width=2, color = 'red', size = 6, legend_label='Average')
base_functions_0.line(x=x_base_interp, y=y_base_interp, line_width=4, color = new_colors[1], legend_label='Interpolated')
base_functions_0 = plot_format(base_functions_0, "Angle", "Amplitude", "top_left", "10pt", "11pt", "8pt")
base_function_plots.append(base_functions_0)

# 4.8 Plot all average base functions in one plot
base_functions_plot.circle(x=x_averaged, y=y_averaged, line_width=2, color = new_colors[0], size = 3, legend_label=f'{file_choice} average')
base_functions_plot.line(x=x_base_interp, y=y_base_interp, line_width=2, color = new_colors[0], legend_label=f'{file_choice} Interpolated')
base_functions_plot = plot_format(base_functions_plot, "Angle", "Amplitude", "top_left", "10pt", "11pt", "8pt")

# 4.9 Save base function
base_function_dict[file_choice] = base_function_merged.copy()

# base_function_plots.append(base_functions_plot)
grid_base = gridplot(children=base_function_plots, ncols=3, merge_tools=False, width = 850, height = 400)
st.bokeh_chart(grid_base)

# %% 5. Minimization iteration to optimize the RMSE
def cost_function_a(params, y, x, chip):
    """
    Generates a 3D plot of the data

    Parameters
    ----------
    params (list): List of parameters to fit
    y (array): Array with the data to fit
    
    Returns
    -------
    rmse (float): Root mean square error of the fit0.
    """
    x0, A0 = params
    x_new = x + x0
    y_modified = A0*chip(x_new)
    mse = np.mean((y - y_modified) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def minimize_procedure(slices_list, starting_minimized, x, chip, values):
    """
    Minimizes the cost function for each slice from the 3D dataset

    Parameters
    ----------
    slices (list): List of slices to optimize
    starting_minimized (DataFrame): DataFrame with initial guesses for the parameters

    Returns
    -------
    dfc (DataFrame): DataFrame with the optimized parameters and RMSE for each slice.
    """
    dfc = pd.DataFrame(columns=["angle", "x0", "difference", "slice", "amplitude", "rmse"])
    for j, sliceb in enumerate(slices_list):
        # Get initial guesses from previous df
        x0 = starting_minimized.loc[sliceb]['x0']
        angle = starting_minimized.loc[sliceb]['angle']
        Abase = starting_minimized.loc[sliceb]['amplitude']
        guess = [angle, Abase]

        # Call minimize function
        ya = values[:, sliceb]
        cost_fn = lambda p:cost_function_a(p, ya, x, chip)
        result = minimize(cost_fn, guess, method=method)
        optimized_parameters = list(result.x)
        x0_opt, A0_opt, = optimized_parameters
        
        # Calculate optimized function
        x_new_opt = x + (x0_opt)
        y_optimized = A0_opt*chip(x_new_opt) 

        # Calculate error
        mse = np.mean((ya - y_optimized) ** 2)
        rmse = np.sqrt(mse)
        row = [offaxis[sliceb], x0_opt, offaxis[sliceb] - x0_opt, sliceb, A0_opt, rmse]
        dfc.loc[j] = row
    return dfc

# 5.1 Define plots
p1 = figure(title=f'a. Rotation angle vs. x0)', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
p2 = figure(title='c. Rotation angle vs. RMSE', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
p3 = figure(title='b. x0 vs. difference (x0-angle)', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
p4 = figure(title='d. Rotation angle vs. amplitude', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
shifted_axis_plot = figure(title='Shifted optosurf axis', width=1100, height=350, tooltips = TOOLTIPS)
base_function_plot = figure(title = 'Base function per iteration', tooltips = TOOLTIPS2, width = 950, height = 600,)

# 5.2 Get initial min_df, starting axis, starting base function and experimental data
slices_amp = np.arange(slice1, slice2+1, step=1)
optimized_final = {}
starting_base = base_function_dict[file_choice].copy()
starting_axis = shifted_axis_dict[file_choice].copy()
starting_minimized = minimized_df_dict[file_choice].copy().set_index('slice')
experimental_data = spots[file_choice].copy()

# 5.3 Update plots for iteration 0
p1.line(x=starting_minimized['angle'], y=starting_minimized['x0'], color=new_colors[0], legend_label=f'Iteration 0', line_width=2)
p2.line(x=starting_minimized['angle'], y=starting_minimized['rmse'], color=new_colors[0], legend_label=f'Iteration 0', line_width=2)
p3.line(x=starting_minimized['angle'], y=starting_minimized['difference'], line_width=2, color=new_colors[0], legend_label=f'Iteration 0', )
p4.line(x=starting_minimized['angle'], y=starting_minimized['amplitude'], line_width=2, color=new_colors[0], legend_label=f'Iteration 0')
shifted_axis_plot.circle(x=starting_axis, y = 0.0, legend_label = 'iteration 0', color = new_colors[0], line_width=2, size=7)
shifted_axis_plot.circle(x=onaxis, y = -0.5, legend_label = 'optosurf axis', color = new_colors[6], line_width=2, size=7)

# 5.4 Calculate average base function
x_averaged, y_averaged = average_base(starting_base, window_size)
pchip_base = PchipInterpolator(x_averaged, y_averaged)
base_function_plot.line(x_averaged, y_averaged, line_width=4, color = new_colors[0], legend_label=f'Iteration 0 interpolated')
base_function_plot.circle(x='xaxis', y='yaxis', source = ColumnDataSource(base_function_merged), size = 4, legend_label=f'Iteration 0 all points', color = new_colors[0])
base_function_plot.triangle(x=x_averaged, y=y_averaged, line_width=2, color = new_colors[0], size = 7, legend_label=f'Iteration 0 average')

# 5.5 Iterate through minimization procedure
for u in range(1,5):
    # 5.6 Calculate the p chip with the starting base function
    x_averaged, y_averaged = average_base(starting_base, window_size)
    pchip_base = PchipInterpolator(x_averaged, y_averaged)

    # 5.7 Call minimization procedure
    min_df = minimize_procedure(slices_amp, starting_minimized, starting_axis, pchip_base, experimental_data)

    # 5.8 Plot rotation angle vs x0 and rotation angle vs RMSE
    p1.line(x=min_df['angle'], y=min_df['x0'], color=new_colors[u], legend_label=f'Iteration {u}', line_width=2)
    p2.line(x=min_df['angle'], y=min_df['rmse'], color=new_colors[u], legend_label=f'Iteration {u}', line_width=2)

    # 5.9 x0 vs difference polynomial fit
    poly = PolynomialFeatures(degree=1)
    x0 = min_df['x0'].values
    difference = min_df['difference'].values
    x0_poly = poly.fit_transform(x0.reshape(-1,1))
    model = LinearRegression().fit(X=x0_poly, y=difference)
    xaxis = np.arange(-15.5, 15.5, 0.1)
    xaxis_poly = poly.transform(xaxis.reshape(-1,1))
    ypredictions = model.predict(xaxis_poly)

    # 5.10 x0 vs difference plot
    p3.line(x=min_df['x0'], y=min_df['difference'], line_width=2, color=new_colors[u], legend_label=f'Iteration {u}', )
    p3.line(x=xaxis, y=ypredictions, line_width=2, color = new_colors[u], dash='dashed', legend_label=f'Iteration {u}')
    
    # 5.11 Calculate shifted axis
    onaxis_poly_temp = poly.transform(starting_axis.reshape(-1,1))
    y_shifted_temp = model.predict(onaxis_poly_temp)
    shifted_axis_temp = starting_axis + y_shifted_temp
    shifted_axis_plot.circle(x=shifted_axis_temp, y=np.zeros(len(onaxis))+0.5*(u), line_width=2, color = new_colors[u], size = 7, legend_label=f'iteration {u}')

    p4.line(x=min_df['angle'], y=min_df['amplitude'], line_width=2, color=new_colors[u], legend_label=f'Iteration {u}')

    # 5.12 Calculate new base function
    x_base_array = []
    y_base_array = []
    slice_array = []
    min_df = min_df.set_index('slice')
    for k, slice in enumerate(slices_amp):
        # 5.13 Get individual slice with 32 points and its corresponding correction factors
        y_temp = experimental_data[:, slice]
        angle = min_df.loc[slice, 'angle']

        # 5.14 Reshift back to center
        new_shifted_axis = shifted_axis_temp + angle
        x_base_array.extend(list(new_shifted_axis))
        y_base_array.extend(list(y_temp))
        slice_array.extend(np.ones(len(new_shifted_axis))*slice)

    # 5.15 Create df with base function
    base_function_df = pd.DataFrame({'xaxis': x_base_array, 'yaxis': y_base_array, 'slice': slice_array})
    base_function_df = base_function_df.sort_values(by='xaxis')

    # 5.16 Merge base function with minimized df
    base_function_merged = base_function_df.merge(min_df, on=['slice', 'slice'])
    
    # 5.17 Update the starting axis and minimized_df
    starting_axis = shifted_axis_temp
    starting_minimized = min_df.copy()
    starting_base = base_function_merged.copy()

    x_averaged, y_averaged = average_base(starting_base, window_size)
    pchip_base = PchipInterpolator(x_averaged, y_averaged)
    base_function_plot.line(x_averaged, y_averaged, line_width=4, color = new_colors[u], legend_label=f'Iteration {u} interpolated')
    base_function_plot.circle(x='xaxis', y='yaxis', source = ColumnDataSource(base_function_merged), size = 4, legend_label=f'Iteration {u} all points', color = new_colors[u])
    base_function_plot.triangle(x=x_averaged, y=y_averaged, line_width=2, color = new_colors[u], size = 7, legend_label=f'Iteration {u} average')

# 5.18 Save final optimized parameters
optimized_axis = starting_axis
optimized_df = starting_minimized
optimized_base = starting_base.copy()

# 5.19 Plot format
p1 = plot_format(p1, "Rotation angle", "x0", "top_left", "10pt", "11pt", "8pt")
p2 = plot_format(p2, "Rotation angle", "RMSE", "top_left", "10pt", "11pt", "8pt")
p3 = plot_format(p3, "x0", "difference", "top_left", "10pt", "11pt", "8pt")
p4 = plot_format(p4, "Rotation angle", "Amplitude", "bottom_left", "10pt", "11pt", "8pt")
base_function_plot = plot_format(base_function_plot, "Angle", "Amplitude", "top_left", "10pt", "11pt", "8pt")
shifted_axis_plot = plot_format(shifted_axis_plot, "Angle", "", "top_left", "10pt", "11pt", "8pt")

grid = gridplot(children=[p1, p2, p3, p4], ncols=2, merge_tools=False, width = 550, height = 400)
st.bokeh_chart(grid)
st.bokeh_chart(shifted_axis_plot)
st.bokeh_chart(base_function_plot)


# %% 6. Roughness comparison

# 6.1 Create editable df
slice_df = pd.DataFrame(
    {"files": files, 
     "slice1": [slice1, slice1, slice1, slice1, slice1],
     "slice2": [slice2, slice2, slice2, slice2, slice2],
     "displacement": [0.0, 0.0, 0.0, 0.0, 0.0],
     "plotb": [True, True, True, True, True],}  
)
edited_df = st.sidebar.data_editor(slice_df)


def cost_function(params, y, x, chip):
    """
    Generates a 3D plot of the data

    Parameters
    ----------
    params (list): List of parameters to fit
    y (array): Array with the data to fit
    
    Returns
    -------
    rmse (float): Root mean square error of the fit0.
    """
    x0, A0 = params
    x_new = x + x0
    y_modified = A0*chip(x_new)

    mse = np.mean((y - y_modified) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# 6.2 Define plots
p1_ann = figure(title=f'a. Rotation angle vs. x0')
p2_ann = figure(title='c. Rotation angle vs. RMSE')
p3_ann = figure(title='b. x0 vs. difference (mechanical angle - x0)')
p4_ann = figure(title='d. Rotation angle vs. amplitude')

# 6.3 Calculate optimized base function pchip
x_average_opt, y_average_opt = average_base(optimized_base, window_size)
pchip_base_optimized = PchipInterpolator(x_average_opt, y_average_opt) 

# 6.4 Plot optimized base function angle vs x0 and angle vs RMSE
p1_ann.line(x=optimized_df['angle'], y=optimized_df['x0'], line_width=2, color = new_colors[0], legend_label = f'Ref {file_choice}')
p2_ann.line(x=optimized_df['angle'], y=optimized_df['rmse'], line_width=2, color = new_colors[0], legend_label = f'Ref {file_choice}')

# 6.5 Polynomial fit of x0 vs difference
x0 = optimized_df['x0'].values
angles = optimized_df['angle'].values
difference = optimized_df['difference'].values
p3_ann.line(x=x0, y=difference, line_width=2, color = new_colors[0], legend_label = f'Ref {file_choice}')

# 6.6 Angle vs amplitude plot
angles = optimized_df['angle'].values
amplitudes = optimized_df['amplitude'].values
p4_ann.line(x=angles, y=amplitudes, line_width=1, color = new_colors[0], legend_label = f'Ref {file_choice}')

# 6.7 Iterate through files for roughness comparison
for z, row in edited_df.iterrows():
    file_ann = row.files
    if file_ann != file_choice:
        if row.plotb == True: # Plot only selected files
            st.write(file_ann)
            minimized_df_ann = pd.DataFrame(columns=["angle", "x0", "difference", "slice", "amplitude", "rmse"])
            data_ann = spots[file_ann]
            offaxis_ann = np.arange(-2048, 2048, 1)*140/4096 + row.displacement
            slices_ann = np.arange(row.slice1, row.slice2, step=1)
            for j, slice in enumerate(slices_ann):
                    
                    # 6.8 Initial guesses
                    x0 = offaxis_ann[slice]
                    Abase = 1.0
                    guess_no_back = [x0, Abase]

                    # 6.9 Call minimization function
                    y = data_ann[:,slice]
                    cost_fn = lambda p:cost_function(p, y, optimized_axis, pchip_base_optimized)

                    result = minimize(cost_fn, guess_no_back, method=method,)
                    optimized_parameters = list(result.x)
                    x0_opt, A0_opt, = optimized_parameters
                    # st.write(x0_opt)
                    # st.write(A0_opt)
                    # 6.10 Calculate optimized function
                    x_new_opt = optimized_axis + (x0_opt)
                    # y_optimized = A0_opt*pchip(x_new_opt) 
                    y_optimized = A0_opt*pchip_base_optimized(x_new_opt)

                    # 6.11 Calculate error
                    mse = np.mean((y - y_optimized) ** 2)
                    rmse = np.sqrt(mse)
                    row2 = [offaxis_ann[slice], x0_opt, offaxis_ann[slice] - x0_opt, slice, A0_opt, rmse]
                    minimized_df_ann.loc[j] = row2
            st.write(minimized_df_ann)
            # 6.12 Rotation angle vs. x0
            # p1_ann.line(x=minimized_df_ann['angle'], y=minimized_df_ann['x0'], line_width=2, color = new_colors[color], legend_label = f'{label_ann[z]} + {file}')
            # st.write(z)
            p1_ann.line(x=minimized_df_ann['angle'], y=minimized_df_ann['x0'], line_width=2, color = new_colors[z+1], legend_label = f'{file_ann}')

            # 6.13 Rotation angle vs RMSE
            p2_ann.line(x=minimized_df_ann['angle'], y=minimized_df_ann['rmse'], line_width=2, color = new_colors[z+1], legend_label = f'{file_ann}')

            # 6.14 Polynomial fit of x0 vs difference
            x0_ann = minimized_df_ann['x0'].values
            angles_ann = minimized_df_ann['angle'].values
            difference_ann = minimized_df_ann['difference'].values
            p3_ann.line(x=x0_ann, y=difference_ann, line_width=2, color = new_colors[z+1], legend_label = f'{file_ann}')

            # 6.15 Rotation angle vs amplitude with interpolation
            angles_ann = minimized_df_ann['angle'].values
            amplitude_ann  = minimized_df_ann['amplitude'].values
            p4_ann.line(x=angles_ann, y=amplitude_ann, line_width=1, color = new_colors[z+1], legend_label = f'{file_ann}')

p1_ann = plot_format(p1_ann, "Rotation angle", "x0", "top_left", "10pt", "11pt", "8pt")
p2_ann = plot_format(p2_ann, "Rotation angle", "RMSE", "top_left", "10pt", "11pt", "8pt")
p3_ann = plot_format(p3_ann, "x0", "difference", "top_left", "10pt", "11pt", "8pt")
p4_ann = plot_format(p4_ann, "Rotation angle", "Amplitude", "top_right", "10pt", "11pt", "8pt")

grid_final = gridplot(children=[p1_ann, p2_ann, p3_ann, p4_ann], ncols=2, merge_tools=False, width = 750, height = 400)
st.bokeh_chart(grid_final)