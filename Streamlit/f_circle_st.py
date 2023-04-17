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

@st.cache_data
def create_matrix(file):
    textfile = open(file, "r")
    line = textfile.readline()
    vals = []
    # 3. Read file lines
    for i in range(0,1024):
        line = textfile.readline().strip()
        semipos = line.find(';')
        subline = line[semipos+1:]
        valsnow = subline.split(' ')
        vals.append(valsnow)
    vals = np.array(vals).T.astype(np.float64) # Generated matrix
    textfile.close()
    return vals

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
        vals = create_matrix(file)
        
        # 4. Normalize values
        normalized_vals = vals[:, 1:] 
        normalized_vals = 10 * vals[:, 1:] / np.max(np.max(vals[:, 1:]))
        
        # 5. Plots
        slice_off_ind = slices[col-1][0]
        slice_on_ind = slices[col-1][1]
        surface = go.Surface(x=offaxis[1:], y=onaxis, z=normalized_vals, colorscale='jet', 
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
                 ('Rotate_Ann7_offaxis_10degscan.txt', 'Rotate_Ann7_onaxis_10degscan.txt', 
                  'Rotate_offaxis_10degscan.txt', 'Rotate_onaxis_10degscan.txt'))

choice2 = st.sidebar.radio("Chose 2nd file",
                 ('Rotate_Ann7_offaxis_10degscan.txt', 'Rotate_Ann7_onaxis_10degscan.txt', 
                  'Rotate_offaxis_10degscan.txt', 'Rotate_onaxis_10degscan.txt'))

# 2. Sliders
st.sidebar.header("Plane slices 1")
slice_off_1 = st.sidebar.slider('Off-axis slice 1', 0, 31, 10, 10)
slice_on_1 = st.sidebar.slider('On-axis slice 1', 0, 1023, 500, 10)

st.sidebar.header("Plane slices 2")
slice_off_2 = st.sidebar.slider('Off-axis slice 2', 0, 31, 10, 1)
slice_on_2 = st.sidebar.slider('On-axis slice 2', 0, 1023, 500, 1)

# Create plots
file1 = "data/f/" + choice1
file2 = "data/f/" + choice2

# 3. Create 3D plots
subplot, subplot2, spots = subplot3d([file1, file2], [[slice_off_1, slice_on_1],[slice_off_2, slice_on_2]])

st.sidebar.header("Camera position")
# x = st.sidebar.slider('x', -5.0, 5.0, 0.5, 0.25)
# y = st.sidebar.slider('y', -5.0, 5.0, 1.25, 0.25)
# z = st.sidebar.slider('z', -5.0, 5.0, 0.5, 0.25)

x = st.sidebar.slider('x', -5.0, 5.0, 1.0, 0.25)
y = st.sidebar.slider('y', -5.0, 5.0, 1.5, 0.25)
z = st.sidebar.slider('z', -5.0, 5.0, 2.75, 0.25)


st.sidebar.header("Plot dimensions")
width = st.sidebar.slider('Plot width', 400, 2000, 800, 20)
height = st.sidebar.slider('Plot height', 400, 2000, 620, 20)
subplot.update_scenes(camera_eye=dict(x=x, y=y, z=z))
subplot.update_layout(width=width, height=height)
st.plotly_chart(subplot)

subplot2.update_layout(width=1000, height=400)
st.plotly_chart(subplot2)

# 4. Aq subplot
titles = list(spots.keys())
titles.extend(["Aq_1", "Aq_2"])
subplot_aq = make_subplots(rows=2, cols=2, 
                             subplot_titles=titles)

col = 1
for key, value in spots.items():
    stds = []
    for i in range(101,1024):
        if i/100 == round(i/100):
            subplot_aq.add_trace(go.Scatter(x=onaxis, y=value[:,i], mode="lines", showlegend=False, line=dict(width=3.5)), row=1, col=col)
            subplot_aq.add_trace(go.Scatter(x=onaxis, y=value[:,i], mode="markers", showlegend=False), row=1, col=col)
            # Histogram reconstruction
            normalized_y = np.multiply(value[:,i], 1)
            hist_2d = np.array([])
            for j, int_point in enumerate(normalized_y):
                round_int_point = round(int(int_point))
                # st.write(round_int_point)
                hist_2d = np.concatenate((hist_2d, np.array([float(j)]*round_int_point)))
            # st.write(hist_2d)
            stddev = np.std(hist_2d)
            stds.append(stddev)
            # st.write(hist_2d)
            # hist, edges = np.histogram(hist_2d, bins=20)
            # subplot_aq.add_trace(go.Histogram(x=hist_2d, nbinsx=10,), row=2, col=col)

        subplot_aq.add_trace(go.Scatter(x=np.arange(0,len(stds)), y=stds, mode="lines", showlegend=False, line=dict(width=3.5)), row=2, col=col)
        subplot_aq.add_trace(go.Scatter(x=np.arange(0,len(stds)), y=stds, mode="markers", showlegend=False, line=dict(width=3.5)), row=2, col=col)
            
            
    col+=1    
subplot_aq.update_layout(width=1000, height=800)
st.plotly_chart(subplot_aq)

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



