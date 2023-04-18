import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from scipy.interpolate import PchipInterpolator
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

offaxis = np.arange(0, 1024, 1)*10/1024
onaxis=np.arange(-15.5,16.5,1)

def subplot3d(files, slices):
    # 1. Define on and off axis as well as subplots
    subplot = make_subplots(rows=1, cols=2, subplot_titles=(choice1, choice2), 
                    specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    subplot2 = make_subplots(rows=1, cols=2, 
                             subplot_titles=("Off-axis slice", "On-axis slice"))
    col = 1

    x = np.linspace(0, 10, 100)
    z = np.linspace(0, 10, 100)
    y = np.linspace(-15.5, 15.5, 100)
    X, Z = np.meshgrid(x, z)
    Y, Z2 = np.meshgrid(y, z)
    count = 0

    spots = {}
    # 2. Read file and create matrix
    for file in files:
        # vals = create_matrix(file)
        vals = np.genfromtxt(file, delimiter=',')
        # vals = vals[:,50:1000]
        
        # 4. Normalize values
        normalized_vals = vals[:, 1:] 
        normalized_vals = 10 * vals[:, 1:] / np.max(np.max(vals[:, 1:]))
        
        # 5. Plots
        slice_off_ind = slices[col-1][0]
        slice_on_ind = slices[col-1][1]
        surface = go.Surface(x=offaxis, y=onaxis, z=normalized_vals, colorscale='jet', 
                   showscale=True)
        slice_off = go.Surface(x=X, y=onaxis[slice_off_ind]*np.ones_like(X), z=Z, opacity=0.3, showscale=False, colorscale='Greys')
        slice_on = go.Surface(x=offaxis[slice_on_ind]*np.ones_like(Y), y=Y, z=Z2, opacity=0.3, showscale=False, colorscale='Greys')

        # Add surface and slice to subplot
        subplot.add_trace(surface, row=1, col=col)
        subplot.add_trace(slice_off, row=1, col=col)
        subplot.add_trace(slice_on, row=1, col=col)
        
        subplot2.add_trace(go.Scatter(x=offaxis, y=vals[slice_off_ind,:], mode="lines", name=f'Off-axis {col}'), row=1, col=1)
        subplot2.add_trace(go.Scatter(x=onaxis, y=vals[:,slice_on_ind], mode="lines", name=f'On-axis {col}'), row=1, col=2)
        col += 1    
        spots[file] = vals
    return subplot, subplot2, spots




# 1. Select files to plot
choice1 = st.sidebar.radio("Chose 1st file",
                 ('Rotate_Ann7_offaxis_10degscan.csv', 'Rotate_Ann7_onaxis_10degscan.csv', 
                  'Rotate_offaxis_10degscan.csv', 'Rotate_onaxis_10degscan.csv'))

choice2 = st.sidebar.radio("Chose 2nd file",
                 ('Rotate_Ann7_offaxis_10degscan.csv', 'Rotate_Ann7_onaxis_10degscan.csv', 
                  'Rotate_offaxis_10degscan.csv', 'Rotate_onaxis_10degscan.csv'))

# 2. Sliders
st.sidebar.header("Plane slices 1")
slice_off_1 = st.sidebar.slider('Off-axis slice 1', 0, 31, 10, 1)
slice_on_1 = st.sidebar.slider('On-axis slice 1', 0, 1023, 500, 10)

st.sidebar.header("Plane slices 2")
slice_off_2 = st.sidebar.slider('Off-axis slice 2', 0, 31, 10, 1)
slice_on_2 = st.sidebar.slider('On-axis slice 2', 0, 1023, 500, 1)

# 3. Create 3D plots
file1 = "data/f/" + choice1
file2 = "data/f/" + choice2
subplot, subplot2, spots = subplot3d([file1, file2], [[slice_off_1, slice_on_1],[slice_off_2, slice_on_2]])

st.sidebar.header("Camera position")
x = st.sidebar.slider('x', -5.0, 5.0, 1.0, 0.25)
y = st.sidebar.slider('y', -5.0, 5.0, 1.5, 0.25)
z = st.sidebar.slider('z', -5.0, 5.0, 2.75, 0.25)


st.sidebar.header("Plot dimensions")
width = st.sidebar.slider('Plot width', 400, 2000, 1200, 20)
height = st.sidebar.slider('Plot height', 400, 2000, 620, 20)
subplot.update_scenes(camera_eye=dict(x=x, y=y, z=z))
subplot.update_layout(width=width, height=height)
st.plotly_chart(subplot)

subplot2.update_layout(width=1000, height=400)
st.plotly_chart(subplot2)

# 4. Plane slices subplot
titles = list(spots.keys())
titles.extend(["Aq_1", "Aq_2"])
subplot_aq = make_subplots(rows=1, cols=2, 
                             subplot_titles=titles)
subplot_aqs = []

new_colors = []
for i in range(42):
        new_colors.append('#2F9C95')
        new_colors.append('#E2DE84')
        new_colors.append('#474973')
        new_colors.append('#6B818C')
        new_colors.append('#C62E65')
        new_colors.append('#4C86A8')

col = 1
for key, value in spots.items():
    subplot = figure(title = key, width = 700, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    stds = []
    count = 0
    for i in range(101,1024):
        if i % 150 == 0:
            subplot.line(x=onaxis, y=value[:,i], line_width=3.5, legend_label=f'{offaxis[i]:.4f}', color=new_colors[count])
            subplot.circle(x=onaxis, y=value[:,i], legend_label=f'{offaxis[i]:.4f}', color = '#65757B')
            # subplot_aq.add_trace(go.Scatter(x=onaxis, y=value[:,i], mode="lines", line=dict(width=3.5), name = f'{offaxis[i]}'), row=1, col=col)
            # subplot_aq.add_trace(go.Scatter(x=onaxis, y=value[:,i], mode="markers", showlegend=False), row=1, col=col)
            
            count += 1
    subplot = plot_format(subplot, "Degrees", "Intensity", "top_right", "10pt", "11pt", "8pt")
    subplot_aqs.append(subplot)    
    col+=1    
# subplot_aq.update_layout(width=1000, height=400)
# st.plotly_chart(subplot_aq)

grid = gridplot(children = subplot_aqs, ncols = 2, merge_tools=False, width = 550, height = 350)
st.bokeh_chart(grid)














            # subplot_aq.add_trace(go.Scatter(x=onaxis, y=value[:,i], mode="lines", showlegend=False, line=dict(width=3.5)), row=1, col=col)
            # subplot_aq.add_trace(go.Scatter(x=onaxis, y=value[:,i], mode="markers", showlegend=False), row=1, col=col)
            
            # Histogram reconstruction
            # normalized_y = np.multiply(value[:,i], 1)
            # hist_2d = np.array([])
            # for j, int_point in enumerate(normalized_y):
            #     round_int_point = round(int(int_point))
            #     hist_2d = np.concatenate((hist_2d, np.array([float(j)]*round_int_point)))
            # stddev = np.std(hist_2d)
            # stds.append(stddev)
       
        # subplot_aq.add_trace(go.Scatter(x=np.arange(0,len(stds)), y=stds, mode="lines", showlegend=False, line=dict(width=3.5)), row=2, col=col)
        # subplot_aq.add_trace(go.Scatter(x=np.arange(0,len(stds)), y=stds, mode="markers", showlegend=False, line=dict(width=3.5)), row=2, col=col)





# subplot_Aq = make_subplots(rows=1, cols=2, 
#                              subplot_titles=list(spots.keys()))










# fig1 = plot3d(file1)
# fig2 = plot3d(file2)
    
# # Create a subplot with two surface plots
# fig = make_subplots(rows=1, cols=2, subplot_titles=(choice1, choice2), 
#                     specs=[[{'type': 'surface'}, {'type': 'surface'}]])

# for trace in fig1['data']:
#     fig.add_trace(trace, row=1, col=1)

# for trace in fig2['data']:
#     fig.add_trace(trace, row=1, col=2)

# fig.update_layout(title='Surface Plots', scene=dict(aspectratio=dict(x=1, y=1, z=0.5), 
#                                                     aspectmode='manual'))
# fig.update_scenes(camera_eye=dict(x=0, y=2.5, z=0.1))

# fig.update_layout(width=width, height=height)
# st.plotly_chart(fig)



# 