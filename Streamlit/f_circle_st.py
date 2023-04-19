import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import Span
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
st.set_page_config(page_title="3D profiles", layout="wide")


def plot_format(plot, xlabel, ylabel, location, size, titlesize, labelsize):
    # x axis format
    plot.xaxis.axis_label = xlabel
    plot.xaxis.axis_label_text_font_size = size
    plot.xaxis.major_label_text_font_size = size
    plot.xaxis.axis_line_color = '#FFFFFF'
    plot.xaxis.major_tick_line_color = '#DAE3F3'
    plot.xaxis.minor_tick_line_color = '#DAE3F3'
    plot.xaxis.major_label_text_color = "#65757B"
    plot.xaxis.axis_label_text_color = "#65757B"
    plot.xgrid.grid_line_color = '#DAE3F3'

    # y axis format
    plot.yaxis.axis_label = ylabel
    plot.yaxis.axis_label_text_font_size = size
    plot.yaxis.major_label_text_font_size = size
    plot.yaxis.axis_line_color = '#FFFFFF'
    plot.yaxis.major_tick_line_color = '#DAE3F3'
    plot.yaxis.minor_tick_line_color = '#DAE3F3'
    plot.yaxis.major_label_text_color = "#65757B"
    plot.yaxis.axis_label_text_color = "#65757B"

    # Legend format
    plot.legend.location = location
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = labelsize
    plot.legend.border_line_width = 3
    plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.0
    plot.legend.background_fill_alpha = 0.0

    # Title format
    plot.title.text_font_size = titlesize
    plot.title.text_font_style = 'normal'
    plot.outline_line_color = '#FFFFFF'
    plot.title.text_color = "#65757B"
    return plot

new_colors = []
for i in range(42):
        new_colors.append('#8e7dbe')
        new_colors.append('#99c1b9')
        new_colors.append('#00b4d8')
        new_colors.append('#f2d0a9')
        new_colors.append('#d88c9a')
        


# offaxis = np.arange(0, 1024, 1)*10/1024
offaxis = np.arange(-512, 512, 1)*20/1024
offaxis = offaxis[::-1]
onaxis=np.arange(-15.5,16.5,1)

# 1. Generate 3d plots
def subplot3d(files, df_slice):
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
    subplot = make_subplots(rows=1, cols=2, subplot_titles=(choice1, choice2), 
                    specs=[[{'type': 'surface'}, {'type': 'scatter'}]])
    x = np.linspace(0, 10, 100)
    z = np.linspace(0, 10, 100)
    y = np.linspace(-15.5, 15.5, 100)
    X, Z = np.meshgrid(x, z)
    Y, Z2 = np.meshgrid(y, z)
    count = 0
    spots = {}
    col = 1

    # b. Read file and create matrix
    for file in files:
        # c. Read the .csv file
        vals = np.genfromtxt(file, delimiter=',')
        
        # d. Normalize values
        normalized_vals = vals[:, 1:] 
        normalized_vals = 10 * vals[:, 1:] / np.max(np.max(vals[:, 1:]))
        
        # e. 3D surface plot
        surface = go.Surface(x=offaxis, y=onaxis, z=normalized_vals, colorscale='jet', 
                   showscale=True)

        # f. Add slice to subplot
        for i, row in df_slice.iterrows():
            slice_off_ind = int(row.slice)
            slice_on = go.Surface(x=offaxis[slice_off_ind]*np.ones_like(Y), y=Y, z=Z2, opacity=0.3, showscale=False, colorscale='Greys')
            subplot.add_trace(slice_on, row=1, col=col)
            subplot.add_trace(go.Scatter(x=onaxis, y=vals[:,slice_off_ind], mode="lines", name=f'On-axis {col}'), row=1, col=2)
        subplot.add_trace(surface, row=1, col=col)
        col += 1    

        # g. Create dictionary
        spots[file] = vals
    return subplot, spots


# 2. Select files to plot
bool_3d = st.sidebar.checkbox("3D plot", value=True)

choice1 = st.sidebar.radio("Chose 1st file",
                 ('Rotate_Ann7_offaxis_10degscan.csv', 'Rotate_Ann7_onaxis_10degscan.csv', 
                  'Rotate_offaxis_10degscan.csv', 'Rotate_onaxis_10degscan.csv'))

choice2 = st.sidebar.radio("Chose 2nd file",
                 ('Rotate_Ann7_offaxis_10degscan.csv', 'Rotate_Ann7_onaxis_10degscan.csv', 
                  'Rotate_offaxis_10degscan.csv', 'Rotate_onaxis_10degscan.csv'))


# 3. Create editable df to select slices to analyse
df = pd.DataFrame({"slice": [100, 250, 400, 750]})
edited_df = st.sidebar.experimental_data_editor(df, num_rows="dynamic")
edited_df["degree"] = offaxis[edited_df["slice"].astype(int)]
st.sidebar.write(edited_df)

# 4. Create 3D plots
file1 = "data/f/" + choice1
file2 = "data/f/" + choice2
subplot, spots = subplot3d([file1], edited_df)

# 5. Define cost function
def cost_function(params, y):
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
    x0, A0, sigma, A1, n, displacement = params
    x_new = onaxis + x0

    # interpolate base function with respect to x_new (32 points)
    y_base_modified = A0*pchip(x_new) 
    # y_base_modified = A0*pchip(onaxis) 
    # calculate background on original axis and with x0
    y_background = supergaussian(x_new, x0+displacement, sigma, A1, n)
    # y_background = supergaussian(onaxis, x0, sigma, A1, n)
    # calculate modified function
    y_modified = y_base_modified + y_background
    if weight_bool:
       mse = np.mean(np.abs(onaxis)*((y - y_modified) ** 2))
       rmse = np.sqrt(mse)  
    else:
        mse = np.mean((y - y_modified) ** 2)
        rmse = np.sqrt(mse)
    # convergence.append(rmse)
    return rmse

# 6. Get base function and define pchip
smooth_df = pd.read_csv("data/f/smooth_df.csv")
x_base = smooth_df['xaxis']
y_base = smooth_df['yaxis']
pchip = PchipInterpolator(x_base, y_base)

# 7. Define super-gaussian function
supergaussian = lambda x, x0, sigma, A1, n: A1 * np.exp(-abs(((x-x0)/sigma))**n)

# 8. Define optimization parameters
methods = ['Powell', 'CG', 'L-BFGS-B', 'SLSQP', 'trust-constr']
method = st.sidebar.radio("Select optimization method", methods)
weight_bool = st.sidebar.checkbox("Weighted optimization", value=True)
limit = st.sidebar.number_input("Limit", value = 50000)

if bool_3d:
    st.sidebar.header("Camera position")
    x = st.sidebar.slider('x', -5.0, 5.0, -0.75, 0.25)
    y = st.sidebar.slider('y', -5.0, 5.0, -2.25, 0.25)
    z = st.sidebar.slider('z', -5.0, 5.0, 1.25, 0.25)

    st.sidebar.header("Plot dimensions")
    width = st.sidebar.slider('Plot width', 400, 2000, 1200, 20)
    height = st.sidebar.slider('Plot height', 400, 2000, 620, 20)
    subplot.update_scenes(camera_eye=dict(x=x, y=y, z=z))
    subplot.update_layout(width=width, height=height)
    st.plotly_chart(subplot)

# 9. Get matrix 32*1024
vals = spots['data/f/Rotate_Ann7_onaxis_10degscan.csv']
optimized_plot = figure(title = 'Optimized plot', width = 700, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
count = 0 

offaxis = np.arange(-512, 512, 1)*20/1024
offaxis = offaxis[::-1]
onaxis=np.arange(-15.5,16.5,1)

edited_df = edited_df.reset_index()
index = ['angle', 'x0', 'Abase', 'sigma', 'Agaussian', 'n', 'displacement', 'error']
columns = ["s_" + str(int(x)) for x in list(edited_df['slice'])]
optimized_df = pd.DataFrame(columns=columns, index=index)
bounds = ((-15, 15), (-0.5, 1.2), (1, 4), (-1000, None), (1, 4), (-3, 3))
            #x0           #Abase    #sigma  #Agaussian    #n       #displacement

# 10. Loop through the selected slices
for i, row in edited_df.iterrows():
        # 11. Define initial guess
        slice_off_ind = int(row.slice)
        x0 = offaxis[slice_off_ind]
        Abase = 1.0
        sigma = 1.0
        Agaussian = 1000.0
        n = 1
        displacement = 0.5
        guess = [x0, Abase, sigma, Agaussian, n, displacement]
        # st.markdown("Initial guess")
        # st.write(guess)
        
        # 12. Call minimize function
        y = vals[:,slice_off_ind]
        cost_fn = lambda p:cost_function(p, y)
        result = minimize(cost_fn, guess, method=method, bounds=bounds)
        # result = minimize(cost_fn, guess, method=method, )
        optimized_parameters = list(result.x)

        # 13. Calculate new optimized functions
        x0_opt, A0_opt, sigma_opt, A1_opt, n_opt, displacement = optimized_parameters
        x_new_opt = onaxis + x0_opt
        y_base_opt = A0_opt*pchip(x_new_opt) 
        y_background_opt = supergaussian(x_new_opt, x0_opt, sigma_opt, A1_opt, n_opt)
        # y_optimized = y_base_opt + y_background_opt
        y_optimized = y_base_opt 

        # 14. Calculate error
        mse = np.mean((y - y_optimized) ** 2)
        rmse = np.sqrt(mse)
        optimized_parameters.append(rmse)

        column = "s_" + str(int(row.slice))
        optimized_parameters.insert(0, row.degree)
        optimized_df[column] = optimized_parameters


        # 14. Plot experimental data
        optimized_plot.line(x=onaxis, y=vals[:,slice_off_ind], line_width=4.5, 
                            legend_label=f'{offaxis[slice_off_ind]:.4f} ({slice_off_ind})', color=new_colors[i])
        optimized_plot.circle(x=onaxis, y=vals[:,slice_off_ind], size = 8,
                              legend_label=f'{offaxis[slice_off_ind]:.4f} ({slice_off_ind})', color = '#65757B')

        # 15. Plot optimized data
        optimized_plot.line(onaxis, y_optimized, line_width = 5, color=new_colors[i+1], 
                            legend_label=f'{slice_off_ind} optimized', dash='dashed')
        optimized_plot.triangle(onaxis, y_optimized, size = 8,
                            legend_label=f'{slice_off_ind} optimized')
        vline = Span(location=0.0, dimension = 'height', line_color='#508AA8', line_width=1)
        optimized_plot.add_layout(vline)

optimized_plot = plot_format(optimized_plot, "Degrees", "Intensity", "top_right", "10pt", "11pt", "8pt")

col1, col2 = st.columns([3, 1.7])
with col1:
    st.bokeh_chart(optimized_plot)    

with col2:
    st.write(optimized_df)
# optimized_plot.line(x=smooth_df['xaxis'], y=smooth_df['yaxis'], line_width=3.5, legend_label='Base function', dash='dashed',)


# for i in range(101,1024):
#         if i % 250 == 0:
#             # Initial guesses
#             # x0 = 

#             # optimized_plot.circle(x=onaxis, y=vals[:,i], legend_label=f'{offaxis[i]:.4f}', color = '#65757B')
#             # optimized_plot.line(x=onaxis, y=vals[:,i], line_width=3.5, legend_label=f'{i}', color=new_colors[count])
#             count += 1


            




# 1. Optimization without background
# 2. find plane where x0 goes to 0
# 3. Compare with real angle axis and have estimation of offset
# 4. Displacemenet
# 5. Calculate error between rotation axis vs x0's
# 6. Plot with respect to new axis
# 7. Plot errors as function of position




            