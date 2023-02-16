# ---
# title: "Wafer roughness plot"
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
import numpy as np
from scipy.interpolate import PchipInterpolator
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Span, Range1d
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
import warnings
import pandas as pd
import altair as alt
import time
warnings.filterwarnings('ignore')
output_notebook()
alt.data_transformers.disable_max_rows()
data = np.loadtxt('data/PT_wafer2_after_d_diodes.dat', delimiter=',')

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
    plot.legend.border_line_alpha = 0.0
    plot.legend.background_fill_alpha = 0.0
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

# ## Calcute Aq, M and I values

# +
lendata = data.shape[0] - 1
angles = (np.linspace(0, (lendata - 1), lendata)) * (data[-1, 29] - data[-1, 28]) * (2 * np.pi / lendata / 360) + (data[-1, 28] + 9.5 + 180) * np.pi / 180
radii = (np.linspace(0, (lendata-1), lendata)) * (data[-1,31]-data[-1,30])/lendata + data[-1, 30]

Aq = []
M = []
intens = []
x = np.arange(1, 33)
datamat = data[:lendata,:] - 1

sx = np.arange(1, 32, 0.2)
for indx in range(lendata):
    # a. Do PChip spline  interpolation
    diodes = datamat[indx,:]
    pchip = PchipInterpolator(x, diodes)
    sy = pchip(sx)
    sy = 100 * sy / np.max(sy)
    c = []

    # b. Concatenation/histogram
    for indx2 in range(len(sx)):
        c.extend([indx2+1]*round(sy[indx2]))

    stddev = np.std(c) / 5
    Aq.append(1.02 * np.exp(1.987 * np.log(stddev) + 0.16))
    M.append((np.mean(c) - 1) / 5 - 15.5)
    intens.append(np.sum(sy))
# -

# ## Non interpolated Plot

# ### Get x/y coordinates

# +
# 1. Get x and y coordinates
xvals = radii * np.cos(angles) * 1e-3
yvals = radii * np.sin(angles) * 1e-3
zvals = Aq

# 2. Clamp values to max of 3
zvalscut = 3
zvalsnew = np.array(zvals)
zvalsnew[zvalsnew > zvalscut] = zvalscut

# 3. Make a df
index = np.arange(len(xvals))
df = pd.DataFrame({
    'index':index,
    'x': xvals*1000,
    'y': yvals*1000,
    'z': zvalsnew,   
})

df.to_csv('data/test.csv', index=False)
# -

# ### Generate plot

# +
#| column: screen-inset-right
# 4. Rank_text
interval = alt.selection_interval()
rank_text = alt.Chart(df).mark_text(dx=20, dy=-5, align='left').encode(
    x=alt.value(10),  # x position of the text
    y=alt.value(10),  # y position of the text
    text=alt.condition(
        interval,
        alt.Text('index:Q', format='.2f'),  # display 'x' column for selected points
        alt.value('')  # display empty string for unselected points
    ),
    color=alt.value('black'), # color of the text
    fontSize = 5
)

# 5. Wafer plot
wafer_plot = alt.Chart(df).mark_circle().encode(
    x='x:Q', y='y:Q',
    color=alt.condition(
        interval,
        alt.Color('z:Q', scale=alt.Scale(scheme='turbo')),
        alt.value('lightgray')
    ),
    tooltip=['x', 'y', 'z', 'index']
).properties(height=400, width=400).add_selection(interval)

# create a table with the selected points
selected_points_table = alt.Chart(df).transform_filter(
    interval
).mark_text().encode(
    x=alt.value(0),
    y=alt.Y('row_number:O', axis=None),
    text='index:Q',
).transform_window(
    row_number='row_number()'
).properties(
    height=400,
    width=50
)

# vertically concatenate the two charts
alt.hconcat(wafer_plot, selected_points_table)


# -

# ## Base function

new_colors = []
for i in range(42):
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')

# +
#| column: screen

# 1. Read the Excel file into a DataFrame
df = pd.read_excel('data/base_function.xlsx', sheet_name=['base', 'M'])

# 2. Split the DataFrame into two separate DataFrames
base_df = df['base']
M_df = df['M'].sort_values(by='M')
sorted_df = pd.DataFrame(columns=['mu','xaxis', 'yaxis', 'colors'])

# 3. Create x axis
xaxis = np.arange(-15.5, 16.5, 1)
plots = []

# 4. Iterate M dataframe
for i, (index, row) in enumerate(M_df.iterrows()):
    # a. Plot raw sampling data
    p = figure(title=str(f'M: {row.M}'), x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 250, height = 150)
    new_axis = xaxis - row.M
    p.line(new_axis, base_df[index], line_color='#9DD9C5', line_width=3)
    p.circle(new_axis, base_df[index], size = 4)
    vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
    p.add_layout(vline)

    # b. Plot format
    p.x_range = Range1d(-7, 7)
    p.yaxis.ticker.desired_num_ticks = 4
    p = plot_format(p, "Degrees", "Intensity", "bottom_left", "8pt", "8pt", "8pt")
    plots.append(p)

    # c. Create dataframe
    sorted_df = sorted_df.append(pd.DataFrame({'mu':[row.M]*32,'xaxis':new_axis, 'yaxis':base_df[index], 'colors':new_colors[0:32]}), ignore_index=True)
    
    # interleaved_plot.circle(new_axis, base_df[index], color=new_colors[i])

grid_raw = gridplot(children = plots, ncols = 6, merge_tools=False)
show(grid_raw)


# +
# 5. Create interleaved plot
interleaved_plot = figure(title='Interleaved points', x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 700, height = 500)
sorted_df_a = sorted_df.sort_values(by='xaxis')

# a. line plot
source = ColumnDataSource(data = sorted_df_a)
interleaved_plot.line(x='xaxis', y='yaxis', source=source, line_width = 6, legend = 'Interleaved curve')

# b. individual points
for (index, row) in sorted_df_a.iterrows():
    interleaved_plot.circle(row.xaxis, row.yaxis, color = row.colors, size = 6)
interleaved_plot.xaxis.ticker.desired_num_ticks = 20
interleaved_plot = plot_format(interleaved_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
show(interleaved_plot)
    
# -



# +

# # Read the Excel file into a DataFrame
# df = pd.read_excel('data/base_function.xlsx', sheet_name=['base', 'M'])

# # Split the DataFrame into two separate DataFrames
# base_df = df['base']
# M_df = df['M'].sort_values(by='M')

# # Create x axis
# xaxis = np.arange(-15.5, 16.5, 1)

# # Iterate M dataframe
# fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(12,8))
# axs = axs.flatten()

# for i, (index, row) in enumerate(M_df.iterrows()):
#     # Calculate shifted axis
#     new_axis = xaxis - row.M
#     # Plot base function
#     axs[i].plot(new_axis, base_df[index], color='#9DD9C5')
#     axs[i].scatter(new_axis, base_df[index], s=15, color='#9DD9C5')
#     # Add vertical line
#     axs[i].axvline(x=0, color='#FEEED9', linewidth=1)
#     # Set x and y axis labels
#     axs[i].set_xlabel('sampling point')
#     axs[i].set_ylabel('intensity')
#     # Set title and tooltip
#     axs[i].set_title(f'M: {row.M}')
#     axs[i].set_xlim(-16, 16)
#     axs[i].set_ylim(0, 40000)
#     # Add grid
#     axs[i].grid(True, linestyle='--', alpha=0.6)
    
# # Adjust layout
# plt.tight_layout()
# plt.show()
# -

# ## Interpolated Plot

# +
zvalscut_hi = 2.04
zvalscut_low = 1.96
zvals = Aq
zvalsnew = np.copy(zvals)
zvalsnew[zvalsnew > zvalscut_hi] = zvalscut_hi
zvalsnew[zvalsnew < zvalscut_low] = zvalscut_low

matindx = []
xaxis = np.arange(-55, 55.5, 0.5) * 1e-3
yaxis = np.arange(-55, 55.5, 0.5) * 1e-3
regmatAq = np.zeros((len(xaxis), len(yaxis)))
nummat = np.zeros((len(xaxis), len(yaxis)))

for xindx in range(len(xaxis)):
    for yindx in range(len(yaxis)):
        d = np.column_stack((xaxis[xindx] - xvals, yaxis[yindx] - yvals))
        e = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        dindx = np.where(e < 0.0008)[0]
        if len(dindx)>0:
            regmatAq[xindx, yindx] = np.mean(zvalsnew[dindx])
        else:
            regmatAq[xindx, yindx] = np.nan

regmatAq[0:2, 0] = [zvalscut_hi, zvalscut_low]

# +
# Create x and y coordinate arrays
x, y = np.meshgrid(xaxis * 1e-3, yaxis * 1e-3)
z = regmatAq
source = pd.DataFrame({'x': x.ravel(),
                     'y': y.ravel(),
                     'z': z.ravel()})

wafer_plot_interpolated = alt.Chart(source).mark_rect().encode(
    x=alt.X('x:O', axis=None),
    y=alt.Y('y:O', axis=None),
    color=alt.Color('z:Q', scale=alt.Scale(scheme='turbo'))
).properties(width=400, height=400).interactive()

wafer_plot_interpolated
# -

# ## Slope plot

# +
zvals = M
df = pd.DataFrame({
    'index':index,
    'x': xvals*1000,
    'y': yvals*1000,
    'z': zvals,   
})

alt.Chart(df).mark_circle().encode(
    x='x:Q', y='y:Q',
    color= alt.Color('z:Q', scale=alt.Scale(scheme='Viridis'),
    
    ),
    tooltip=['x', 'y', 'z', 'index']
).properties(height=400, width=400).interactive()

# -

# ## Plot - angle

# +
zvals = M
zvalscut=0.2
zvalsnew= np.array(zvals)
zvalsnew[np.abs(zvalsnew) > zvalscut] = zvalscut

df = pd.DataFrame({
    'index':index,
    'x': xvals*1000,
    'y': yvals*1000,
    'z': zvalsnew,   
})

alt.Chart(df).mark_circle().encode(
    x='x:Q', y='y:Q',
    color= alt.Color('z:Q', scale=alt.Scale(scheme='Turbo'),
    
    ),
    tooltip=['x', 'y', 'z', 'index']
).properties(height=400, width=400).interactive()

