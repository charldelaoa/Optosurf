import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def plot3d(file):
    textfile = open(file, "r")
    line = textfile.readline()

    vals = []
    count = 0
    onaxis=np.arange(-15.5,16.5,1)

    for i in range(0,1024):
        line = textfile.readline().strip()
        semipos = line.find(';')
        subline = line[semipos+1:]
        valsnow = subline.split(' ')
        vals.append(valsnow)

    vals = np.array(vals).T.astype(np.float64)
    textfile.close()

    offaxis = np.arange(0, 1024, 1)*10/1024
    onaxis=np.arange(-15.5,16.5,1)
    normalized_vals = vals[:, 1:] 

    normalized_vals = 10 * vals[:, 1:] / np.max(np.max(vals[:, 1:]))
    plot = go.Figure(data=[go.Surface(x=offaxis[1:], y=onaxis, z=normalized_vals, 
    colorscale='jet', showscale=True, cauto=False,)])
    plot.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    plot.update_layout(scene=dict(aspectratio=dict(x=1.5, y=1, z=1), aspectmode='manual'))
    plot.update_layout(width=width, height=height)

    return plot

def subplot3d(files):
    subplot = make_subplots(rows=1, cols=2, subplot_titles=(choice1, choice2), 
                    specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    
    subplot2 = make_subplots(rows=1, cols=2, subplot_titles=(choice1, choice2, "Line Plot 1", "Line Plot 2"), 
                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])
    col = 1
    for file in files:
        textfile = open(file, "r")
        line = textfile.readline()

        vals = []
        count = 0
        onaxis=np.arange(-15.5,16.5,1)

        for i in range(0,1024):
            line = textfile.readline().strip()
            semipos = line.find(';')
            subline = line[semipos+1:]
            valsnow = subline.split(' ')
            vals.append(valsnow)
            if i/100 == round(i/100):
                subplot2.add_trace(go.Scatter(x=onaxis, y=valsnow, mode="lines", showlegend=False, line=dict(width=3.5)), row=1, col=col)
                subplot2.add_trace(go.Scatter(x=onaxis, y=valsnow, mode="markers", showlegend=False), row=1, col=col)
        vals = np.array(vals).T.astype(np.float64)
        textfile.close()

        offaxis = np.arange(0, 1024, 1)*10/1024
        onaxis=np.arange(-15.5,16.5,1)
        normalized_vals = vals[:, 1:] 
        
        normalized_vals = 10 * vals[:, 1:] / np.max(np.max(vals[:, 1:]))
        
        subplot.add_trace(
        go.Surface(x=offaxis[1:], y=onaxis, z=normalized_vals, colorscale='jet', 
                   showscale=True, cauto=False), row=1, col=col)
        
        # subplot.update_layout(scene=dict(aspectratio=dict(x=1.5, y=1, z=1), aspectmode='manual'))
        col += 1    
    return subplot, subplot2


# Select files to plot
choice1 = st.sidebar.radio("Chose 1st file",
                 ('Rotate_Ann7_offaxis_10degscan.txt', 'Rotate_Ann7_onaxis_10degscan.txt', 
                  'Rotate_offaxis_10degscan.txt', 'Rotate_onaxis_10degscan.txt'))

choice2 = st.sidebar.radio("Chose 2nd file",
                 ('Rotate_Ann7_offaxis_10degscan.txt', 'Rotate_Ann7_onaxis_10degscan.txt', 
                  'Rotate_offaxis_10degscan.txt', 'Rotate_onaxis_10degscan.txt'))

width = st.sidebar.slider('Plot width', 400, 2000, 800, 20)
height = st.sidebar.slider('Plot height', 400, 2000, 800, 20)

# Create plots
file1 = "data/f/" + choice1
file2 = "data/f/" + choice2

subplot, subplot2 = subplot3d([file1, file2])
x = st.sidebar.slider('x', 0.0, 5.0, 0.0, 0.5)
y = st.sidebar.slider('y', 0.0, 5.0, 2.5, 0.5)
z = st.sidebar.slider('z', 0.0, 5.0, 0.5, 0.5)

subplot.update_scenes(camera_eye=dict(x=x, y=y, z=z))
# subplot.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                     highlightcolor="limegreen", project_z=True))
subplot.update_layout(width=width, height=height)
st.plotly_chart(subplot)

subplot2.update_layout(width=1000, height=500)
st.plotly_chart(subplot2)


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



