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
from scipy.optimize import minimize, curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

offaxis = np.arange(-512, 512, 1)*20/1024 + 1.82
onaxis=np.arange(-15.5,16.5,1)

# 1. Generate 3d plots
def subplot3d(files, df_slice, slices):
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

        for slice in slices:
            slice_off_ind = slice
            slice_on = go.Surface(x=offaxis[slice]*np.ones_like(Y), y=Y, z=Z2, opacity=0.6, showscale=False, colorscale='Viridis')
            subplot.add_trace(slice_on, row=1, col=col)
        subplot.add_trace(surface, row=1, col=col)
        col += 1    

        # g. Create dictionary
        spots[file] = vals
    return subplot, spots

# 2. Select files to plot
bool_3d = st.sidebar.checkbox("3D plot", value=True)

container = st.sidebar.container()
container.header("Select files to plot")
choice1 = container.radio("Chose 1st file",
                 ('Rotate_Ann7_offaxis_10degscan.csv', 'Rotate_Ann7_onaxis_10degscan.csv', 
                  'Rotate_offaxis_10degscan.csv', 'Rotate_onaxis_10degscan.csv'))

choice2 = container.radio("Chose 2nd file",
                 ('Rotate_Ann7_offaxis_10degscan.csv', 'Rotate_Ann7_onaxis_10degscan.csv', 
                  'Rotate_offaxis_10degscan.csv', 'Rotate_onaxis_10degscan.csv'))

container.header("Select a range of slices to analyse/minimise")
slice_1 = container.number_input('Starting slice', min_value=0, max_value=1023, value=250)
slice_2 = container.number_input('Finish slice', min_value=0, max_value=1023, value=750)
slice_step = container.number_input('Slice step', min_value=0, max_value=50, value=1)
polynomial = container.number_input('Polynomial degree', min_value=1, max_value=50, value=3)
container.markdown(f'{offaxis[slice_2]:.2f} to {offaxis[slice_1]:.2f} degrees')

# 3. Create editable df to select slices to analyse
st.sidebar.header("Select individual slices")
df = pd.DataFrame({"slice": [50, 250, 500, 750, 1000]})
edited_df = st.sidebar.experimental_data_editor(df, num_rows="dynamic")
edited_df["degree"] = offaxis[edited_df["slice"].astype(int)]
st.sidebar.write(edited_df)

# 4. Create 3D plots
file1 = "data/f/" + choice1
file2 = "data/f/" + choice2
subplot, spots = subplot3d([file1], edited_df, [slice_1, slice_2])

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
    
vals = spots['data/f/Rotate_Ann7_onaxis_10degscan.csv']
minimized_df = pd.read_csv('data/f/minimized.csv', index_col = 'slice')
shifted_axis = pd.read_csv('data/f/shifted_axis.csv').values

optimized_plot = figure(title = 'Optimized plot', width = 700, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
count = 0 

smooth_df = pd.read_csv("data/f/smooth_df.csv")
x_base = smooth_df['xaxis']
y_base = smooth_df['yaxis']
pchip = PchipInterpolator(x_base, y_base)

for i, row in edited_df.iterrows():
    slice = int(row.slice)
    y = vals[:, slice]

    A0_opt = minimized_df.loc[slice, 'amplitude']
    x0_opt = minimized_df.loc[slice, 'x0']
    x_new_opt = onaxis + x0_opt
    y_optimized = A0_opt*pchip(x_new_opt)

    optimized_plot.line(x=onaxis, y=y, line_width=4.5, 
                            legend_label=f'{offaxis[slice]:.4f} ({slice})', color=new_colors[i])
    optimized_plot.circle(x=onaxis, y=vals[:,slice], size = 8,
                            legend_label=f'{offaxis[slice]:.4f} ({slice})', color = '#65757B')

    optimized_plot.line(onaxis, y_optimized, line_width = 5, color=new_colors[i+1], 
                            legend_label=f'{slice} optimized', dash='dashed')
    optimized_plot.triangle(onaxis, y_optimized, size = 8,
                            legend_label=f'{slice} optimized')
    vline = Span(location=0.0, dimension = 'height', line_color='#508AA8', line_width=1)
    optimized_plot.add_layout(vline)
    
optimized_plot = plot_format(optimized_plot, "Degrees", "Intensity", "top_right", "10pt", "11pt", "8pt")       
bool_optimized = st.checkbox("Show optimized plot", value=True)
if bool_optimized:
    st.bokeh_chart(optimized_plot)

col3, col4 = st.columns([3, 1.7])

def linear_function(x, m , b):
    return m*x + b

angle = minimized_df['angle']
x0 = minimized_df['x0']
difference = minimized_df['difference']

# angle vs. x0 linear fit
params, covariance = curve_fit(linear_function, angle, x0)   
slope = params[0]
intercept = params[1]
angle = np.arctan(slope)
angle_degrees = np.degrees(angle)

# x0 vs difference fit
poly = PolynomialFeatures(degree=polynomial)
x0 = minimized_df['x0'].values
X_poly = poly.fit_transform(x0.reshape(-1,1))
model = LinearRegression().fit(X_poly, difference)
xaxis = np.arange(-15.5, 15.5, 0.1)
xaxis_poly = poly.transform(xaxis.reshape(-1,1))
ypredictions = model.predict(xaxis_poly)

# Estimate angle differences
onaxis_poly = poly.transform(onaxis.reshape(-1,1))
y_shifted = model.predict(onaxis_poly)
shifted_axis = onaxis - y_shifted

# shifted_axis_df = pd.DataFrame(shifted_axis, columns=['shifted_axis'])
# shifted_axis_df.to_csv('data/f/shifted_axis.csv', index=False)

bool_grid = st.checkbox("Show grid", value=True)

with col3:
    p1 = figure(title=f'Angle vs. x0 (x0 = {slope:.2f}*angle {intercept:.2f})')
    p1.line(x=minimized_df['angle'], y=minimized_df['x0'], line_width=2, color = new_colors[0])
    p1.xaxis.ticker.desired_num_ticks = 10
    p1 = plot_format(p1, "Angle", "x0", "top_right", "10pt", "11pt", "8pt")\
    
    p2 = figure(title='Slice vs. RMSE')
    p2.line(x=minimized_df['angle'], y=minimized_df['rmse'], line_width=2, color = new_colors[1])
    p2.xaxis.ticker.desired_num_ticks = 10
    p2 = plot_format(p2, "Angle", "RMSE", "top_right", "10pt", "11pt", "8pt")

    p3 = figure(title='x0 vs. difference (x0-angle)')
    p3.line(x=minimized_df['x0'], y=minimized_df['difference'], line_width=2, color = new_colors[2])
    p3.line(x=xaxis, y=ypredictions, line_width=2, color = new_colors[4], dash='dashed')
    p3.xaxis.ticker.desired_num_ticks = 10
    p3 = plot_format(p3, "x0", "difference", "top_right", "10pt", "11pt", "8pt")

    p4 = figure(title='Angle vs. amplitude')
    p4.line(x=minimized_df['angle'], y=minimized_df['amplitude'], line_width=2, color = new_colors[4])
    p4.xaxis.ticker.desired_num_ticks = 10
    p4 = plot_format(p4, "Angle", "amplitude", "top_right", "10pt", "11pt", "8pt")
    
    grid = gridplot(children=[p1, p2, p3, p4], ncols=2, merge_tools=False, width = 350, height = 340)

    p5 = figure(title='Shifted optosurf axis', width=1000, height=350, y_range=(-0.2, 0.9))
    p5.circle(x=shifted_axis, y=np.zeros(len(onaxis))+0.5, line_width=2, color = new_colors[6], size = 7, legend_label="Shifted axis")
    p5.circle(x=onaxis, y=np.zeros(len(onaxis)), line_width=2, color = new_colors[5], size = 7, legend_label="Original axis")
    p5.xaxis.ticker.desired_num_ticks = 40
    p5 = plot_format(p5, "Slice", "angle", "top_right", "10pt", "11pt", "8pt")
    
    if bool_grid:
        st.bokeh_chart(grid)
        st.bokeh_chart(p5)

with col4:
    if bool_grid:
        st.dataframe(minimized_df, height=700)

# Shifted plots
shifted_plot = figure(title = 'Shifted plot', width = 1000, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
for i, row in edited_df.iterrows():
    slice_off_ind = int(row.slice)
    y = vals[:,slice_off_ind]
    shifted_plot.line(x=onaxis, y=y, line_width=4.5, 
                            legend_label=f'Original: {offaxis[slice_off_ind]:.4f} ({slice_off_ind})', color=new_colors[i], dash='dashed', alpha = 0.5)
    shifted_plot.circle(x=onaxis, y=y, size = 8,
                            legend_label=f'Original: {offaxis[slice_off_ind]:.4f} ({slice_off_ind})', color = '#65757B')
    shifted_plot.line(x=shifted_axis, y=y, line_width=4.5, 
                            legend_label=f'Shifted: {offaxis[slice_off_ind]:.4f} ({slice_off_ind})', color=new_colors[i],)
    shifted_plot.triangle(x=shifted_axis, y=y, size = 8,
                            legend_label=f'Shifted: {offaxis[slice_off_ind]:.4f} ({slice_off_ind})', color = '#65757B')
shifted_plot = plot_format(shifted_plot, "Angle", "Amplitude", "top_right", "10pt", "11pt", "8pt")                            
bool_shifted = st.checkbox("Show shifted plot", value=True)
if bool_shifted:
    st.bokeh_chart(shifted_plot)


# Base function
st.header("Base function")
with st.expander('Analysis'):
    base_1 = st.number_input('Starting base function slice', min_value=0, max_value=1023, value=20)
    base_2 = st.number_input('Finish base function slice', min_value=0, max_value=1023, value=1000)
    base_step = st.number_input('Base function step', min_value=0, max_value=1023, value=1)
    base_plot = figure(title = 'Base function', width = 1300, height = 500, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    base_plot_2 = figure(title = 'Base function clusters', width = 1300, height = 500, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    slices_base = np.arange(base_1, base_2+1, step=base_step)
    st.write(f'Range from {offaxis[base_1]:.3f} to  {offaxis[base_2]:.3f}')
    left = st.number_input('Left angle', min_value=-15.0, max_value=15.0, value=-5.0)
    right = st.number_input('Right angle', min_value=-15.0, max_value=15.0, value=5.0)


# Populate base function by shifting with respect to zero
x_base = []
y_base = []
slice_array = []

for k, slice in enumerate(slices_base):
    y = vals[:,slice]
    x0 = minimized_df.loc[slice, 'x0']
    amp = minimized_df.loc[slice, 'amplitude']
    shifted_axis_2 = shifted_axis + offaxis[slice]
    # y = y*amp
    # shifted_axis_2 = shifted_axis + x0
    # shifted_axis_2 = onaxis + offaxis[slice]
    # shifted_axis_2 = shifted_axis + x0
    # shifted_axis_2 = shifted_axis + x0
    x_base.extend(list(shifted_axis_2))
    y_base.extend(list(y))
    slice_array.extend(np.ones(len(shifted_axis_2))*slice)

base_function_df = pd.DataFrame({'xaxis': x_base, 'yaxis': y_base, 'slice': slice_array})
base_function_df = base_function_df.sort_values(by='xaxis')

x_base, y_base = zip(*sorted(zip(x_base, y_base)))
x_base = np.array(x_base)
y_base = np.array(y_base)

mask = (x_base >= left) & (x_base <= right)
x_filtered = x_base[mask]
y_filtered = y_base[mask]
window_size = 0.06
x_averaged = []
y_averaged = []

# loop through x_filtered with a window size 
for i in np.arange(np.min(x_filtered), np.max(x_filtered), window_size):
    # get the indices of the points within the current window
    indices = np.where((x_filtered >= i) & (x_filtered < i + window_size))[0]
    if len(indices) == 0:
        continue
    # calculate the average x and y values for the points in the current window
    x_avg = np.mean(x_filtered[indices])
    y_avg = np.mean(y_filtered[indices])
    # append the averaged x and y values to the lists
    x_averaged.append(x_avg)
    y_averaged.append(y_avg)

x_averaged = np.array(x_averaged)
y_averaged = np.array(y_averaged)

# perform spline interpolation on the averaged x and y values
# f = interp1d(x_averaged, y_averaged, kind='cubic')
f = PchipInterpolator(x_averaged, y_averaged)
x_interp = np.linspace(x_averaged[0], x_averaged[-1], num=5000)
y_interp = f(x_interp)

base_plot_2.circle(x=x_filtered, y=y_filtered, size=5.5, color = new_colors[1], legend_label='Original')
base_plot_2.line(x=x_interp, y=y_interp, line_width=5.5,  
                 color = new_colors[5], legend_label='Interpolation')
base_plot_2.circle(x=x_averaged, y=y_averaged, size=5.5, color = '#EC5766', legend_label='Averaged')

base_plot_2 = plot_format(base_plot_2, "Angle", "Amplitude", "top_right", "10pt", "11pt", "8pt")
# st.bokeh_chart(base_plot_2)

## sections plot
new_colors = []
for i in range(42):
        new_colors.append('#99c1b9')
        new_colors.append('#8e7dbe')
        new_colors.append('#00b4d8')
        new_colors.append('#38A3A5')
        new_colors.append('#d88c9a')
        new_colors.append('#F283B6')
        new_colors.append('#f2d0a9')

base_function_df = base_function_df
sections_df = pd.DataFrame({'start': [20], 'end': [25]})
sections_edited_df = st.sidebar.experimental_data_editor(sections_df, num_rows="dynamic")

base_sections = figure(title = 'Base function sections', width = 1300, height = 800, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
# st.write(base_function_df)
# group base_function_df by slice

for z, row in sections_edited_df.iterrows():
    slices = np.arange(int(row['start']), int(row['end']+1), step=1)
    amplitudes = minimized_df.loc[slices]['amplitude']
    # st.write(amplitudes)
    mask = base_function_df['slice'].isin(slices)
    sliced_df = base_function_df.loc[mask]
    sliced_df.sort_values(by=['slice', 'xaxis'], inplace=True)
    # st.write(sliced_df)
    for k, amp in amplitudes.iteritems():
        internal_slice = int(k)
        sliced_df_2 = sliced_df[sliced_df['slice'] == internal_slice]
        sliced_df_2['yaxis'] = sliced_df_2['yaxis'] / amp
        base_sections.triangle(x=sliced_df_2['xaxis'], y=sliced_df_2['yaxis'], size=8.5, 
                               color = new_colors[z], legend_label=f'{int(row.start)} to {int(row.end)} amp')

        # st.write(row2)
    # base_sections.circle(x=sliced_df['xaxis'], y=sliced_df['yaxis'], size=5.5, color = new_colors[z], legend_label=f'{int(row.start)} to {int(row.end)}')

base_sections.xaxis.ticker.desired_num_ticks = 20
base_sections = plot_format(base_sections, "Angle", "Amplitude", "top_right", "10pt", "11pt", "8pt")
st.bokeh_chart(base_sections)
st.bokeh_chart(base_plot_2)

st.write(minimized_df)
#         a = base_function_df.groupby('slice')
#         st.write(a)
        # y = vals[:, slice]

    # y = vals[:,slice]
    # x0 = minimized_df.loc[slice, 'x0']
    # amp = minimized_df.loc[slice, 'amplitude']
    # shifted_axis_2 = shifted_axis + offaxis[slice]
























