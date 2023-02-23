# ---
# title: "Wafer roughness plot"
# format: 
#   html:
#     code-fold: true
#     code-tools: true
#     theme: slate
#     toc: true
#     toc-location: right
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
#     display_name: pandasapps
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

# ## Base function

# ### Different $\mu$ sampling points 

#| echo: false
new_colors = []
for i in range(42):
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')

# +
#| column: page

# 1. Read the Excel file into a DataFrame
df = pd.read_excel('data/base_function.xlsx', sheet_name=['base', 'M'])

# 2. Split the DataFrame into two separate DataFrames
base_df = df['base']
M_df = df['M'].sort_values(by='M')
# M_df = M_df[~M_df.isin([-0.002, 0.003]).any(axis=1)]
sorted_df = pd.DataFrame(columns=['mu','xaxis', 'yaxis', 'colors'])

# 3. Create x axis
xaxis = np.arange(-15.5, 16.5, 1)
plots = []

# 4. Iterate M dataframe
for i, (index, row) in enumerate(M_df.iterrows()):
    # a. Plot raw sampling data
    p = figure(title=str(f'M: {row.M}'), x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 250, height = 200)
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
    
grid_raw = gridplot(children = plots, ncols = 6, merge_tools=False)
show(grid_raw)

# -

# ### Base function smoothing and interpolation

# +
#| column: page
# 5. Create interleaved plots
interleaved_plot = figure(title='Interleaved base function', x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 700, height = 500)
smooth_plot = figure(title='Smooth base function', x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 700, height = 500)
interpolated_plot = figure(title='Inteporlated base function points', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 700, height = 500)

# a. Define base_function and smooth df's
# diff = base_function_df['xaxis'].diff()
# smooth_df = base_function_df[(diff >= 0.01) | (diff.isna())]
# smooth_df = smooth_df.iloc[1:]
base_function_df = sorted_df.sort_values(by='xaxis').reset_index(drop=True)
smooth_df = pd.DataFrame(data={}, columns=['xaxis', 'yaxis', 'colors'])
xoutindx=0

for aveindex in range(1, len(base_function_df)):
    if (base_function_df.loc[aveindex, 'xaxis'] - base_function_df.loc[aveindex-1, 'xaxis']) < 0.01:
        smooth_df.loc[xoutindx, 'xaxis'] = (base_function_df.loc[aveindex, 'xaxis'] + base_function_df.loc[aveindex-1, 'xaxis'])/2
        smooth_df.loc[xoutindx, 'yaxis'] = (base_function_df.loc[aveindex, 'yaxis'] + base_function_df.loc[aveindex-1, 'yaxis'])/2
        smooth_df.loc[xoutindx, 'colors'] = base_function_df.loc[aveindex, 'colors']
    else:
        xoutindx += 1
        smooth_df.loc[xoutindx, 'xaxis'] = base_function_df.loc[aveindex, 'xaxis']
        smooth_df.loc[xoutindx, 'yaxis'] = base_function_df.loc[aveindex, 'yaxis']
        smooth_df.loc[xoutindx, 'colors'] = base_function_df.loc[aveindex, 'colors']


# b. Create non-smooth and smoot curve
interleaved_plot = figure(title='Interleaved base function', x_axis_label='sampling point', y_axis_label='intensity', tooltips=[("index", "$index"), ("(x,y)", "($x, $y)")],
                          width=700, height=500)
smooth_plot = figure(title='Smooth base function', x_axis_label='sampling point', y_axis_label='intensity', tooltips=[("index", "$index"), ("(x,y)", "($x, $y)")],
                     width=700, height=500)

# c. Plot points
for (plot, df, legend, color) in [(interleaved_plot, base_function_df, 'Non-smooth base function', '#9DC3E6'), (smooth_plot, smooth_df, 'Smooth base function', '#9D6C97')]:
    # individual points
    plot.circle(df.xaxis, df.yaxis, color=df.colors, size=6)

    # smooth curve
    plot.line(df['xaxis'], df['yaxis'], line_width=4, legend=legend, color=color)

    # format
    plot.xaxis.ticker.desired_num_ticks = 15
    plot.y_range = Range1d(0, 45000)
    plot = plot_format(plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")

# d. Interpolation
x_base = np.arange(-15.5, 15.5001, 0.001)
pchip = PchipInterpolator(smooth_df['xaxis'], smooth_df['yaxis'])
y_base = pchip(x_base)

interpolated_plot.line(x=smooth_df['xaxis'], y=smooth_df['yaxis'], line_width = 5, legend = 'Smooth base function', color = '#9D6C97')
interpolated_plot.line(x_base, y_base, line_width = 5, color = '#9DD9C5', legend = 'Interpolated base function')
interpolated_plot.xaxis.ticker.desired_num_ticks = 15
interpolated_plot.y_range = Range1d(-1000, 45000)

interpolated_plot = plot_format(interpolated_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
base_function_grid = gridplot(children=[interleaved_plot, smooth_plot, interpolated_plot], ncols=3, merge_tools=False, width=420, height=380)
show(base_function_grid)
# -

# ## Experimental rough data

# Once the base function has been reconstructed from smooth experimental data, the next step is to use such function as a reference to compare it with rough data. This is illustrated in @fig-1. Notice how the base function (green) is sharper and with a higher amplitude than the experimental data (blue). 
#
# The problem to be solved is to calculate parameters to modify the base function in order to obtain the amplitude difference as well as the tails as observed from the rough data.

# ::: {#fig-1}
#
# ![](images/c/Fig_1.png)
#
# Obtain parameters to go from the base 'smooth' function to a rougher curve with decreased amplitude and higher side tails
# :::

# +
#| column: screen-inset-right
from bokeh.palettes import Set3
# 1. Import data
rough_df = pd.read_excel('data/rough_samples.xlsx')
source_rough = ColumnDataSource(rough_df)

# # 2. Create plot
rough_plots = []
color_palette = Set3[len(rough_df.columns[1:])+2]

# a. iterate over the columns and add a line for each one
for i, col in enumerate(rough_df.columns[1:]):
    rough_plot = figure(title = str(col), x_axis_label='xaxis', y_axis_label='yaxis', width = 420, height = 380, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    rough_plot.line(x_base, y_base, line_width=4, color = '#9D6C97', legend_label = 'base function')
    rough_plot.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4)
    rough_plot.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    rough_plot = plot_format(rough_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
    rough_plots.append(rough_plot)

# #9D6C97')
#         new_colors.append('#9DC3E6')
#         new_colors.append('#9DD9C5')
grid_rough = gridplot(children = rough_plots, ncols = 3, merge_tools=False)
show(grid_rough)
# xaxis = np.arange(-15.5, 16.5, 1) - M_df.loc[0,'M']
# rough_plot.line(xaxis, base_df[0], line_width=4, color = color_palette[i+1], legend = 'M = 0.001')
# rough_plot.circle(xaxis, base_df[0], line_width=4, color = color_palette[i+1], legend = 'M = 0.001')


# +
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
import numpy as np
from scipy.special import voigt_profile


# Define the functions to be plotted
def voigt(x, x0, sigma, gamma):
    return voigt_profile(x-x0, sigma, gamma)

def pseudo_voigt(x, x0, sigma, alpha):
    return (1 - alpha) * np.exp(-((x-x0)/sigma)**2/2) + alpha/(1 + ((x-x0)/sigma)**2)

def cauchy(x, x0, gamma):
    return gamma / (np.pi * (gamma**2 + (x-x0)**2))

def student_t(x, x0, sigma, nu):
    return (1/(sigma*np.sqrt(np.pi*nu/2))) * (1 + ((x-x0)/sigma)**2/nu)**(-(nu+1)/2)

def pearson(x, x0, sigma, skew, kurtosis):
    return (1/(sigma*np.sqrt(2*np.pi))) * ((1 + ((x-x0)/sigma)**2)**(-(skew+1)/(2*kurtosis)))

def sigmoid(x, x0, k):
    return 1/(1 + np.exp(-k*(x-x0)))

# Define the functions to be plotted
def gaussian(x, x0, sigma):
    return np.exp(-((x-x0)/sigma)**2/2)

def lorentzian(x, x0, gamma):
    return 1/(1 + ((x-x0)/gamma)**2)

def voigt(x, x0, sigma, gamma):
    return (1-gamma)*gaussian(x, x0, sigma) + gamma*lorentzian(x, x0, sigma)

def pseudo_voigt(x, x0, sigma, alpha):
    return (1 - alpha) * gaussian(x, x0, sigma) + alpha*lorentzian(x, x0, sigma)

def cauchy(x, x0, gamma):
    return gamma / (np.pi * (gamma**2 + (x-x0)**2))

def student_t(x, x0, sigma, nu):
    return (1/(sigma*np.sqrt(np.pi*nu/2))) * (1 + ((x-x0)/sigma)**2/nu)**(-(nu+1)/2)

def pearson(x, x0, sigma, skew, kurtosis):
    return (1/(sigma*np.sqrt(2*np.pi))) * ((1 + ((x-x0)/sigma)**2)**(-(skew+1)/(2*kurtosis)))

def sigmoid(x, x0, k):
    return 1/(1 + np.exp(-k*(x-x0)))

def squared_cosine(x, x0, c):
    return np.where(np.abs(x-x0) <= c, 0.5*(1 + np.cos(np.pi*(x-x0)/c)), 0)


# xaxis = np.arange(-15.5, 16.5, 1)
# Generate some x and y data for each function
x = np.linspace(-15.5, 16.5, 1000)
y1 = gaussian(x, 0, 1)
y2 = lorentzian(x, 0, 1)
y3 = voigt(x, 0, 1, 0.5)
y4 = pseudo_voigt(x, 0, 1, 0.5)
y5 = cauchy(x, 0, 1)
y6 = student_t(x, 0, 1, 5)
y7 = pearson(x, 0, 1, 0, 1)
y8 = sigmoid(x, 0, 1)
y9 = squared_cosine(x, 0, 2)
plots = []
# Create the Bokeh figures for each function
p1 = figure(title="Gaussian function")
p1.line(x, y1)
plots.append(p1)

p2 = figure(title="Lorentzian function")
p2.line(x, y2)
plots.append(p2)

p3 = figure(title="Voigt function")
p3.line(x, y3)
plots.append(p3)

p4 = figure(title="Pseudo-Voigt function")
p4.line(x, y4)
plots.append(p4)

p5 = figure(title="Cauchy distribution")
p5.line(x, y5)
plots.append(p5)

p6 = figure(title="Student's t-distribution")
p6.line(x, y6)
plots.append(p6)

p7 = figure(title="Pearson distribution")
p7.line(x, y7)
plots.append(p7)

p8 = figure(title="Sigmoid function")
p8.line(x, y8)
plots.append(p8)

p9 = figure(title="Squared cosine function")
p9.line(x, y9)
plots.append(p9)

grid_maths = gridplot(children = plots, ncols = 3, merge_tools=False, width = 450, height = 320)
show(grid_maths)

# +
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
import numpy as np

# Define the functions to be plotted
functions = [
    ("Gaussian", lambda x, x0, sigma: np.exp(-((x-x0)/sigma)**2/2), (0, 1)),
    ("Lorentzian", lambda x, x0, gamma: 1/(1 + ((x-x0)/gamma)**2), (0, 1)),
    ("Voigt", lambda x, x0, sigma, gamma: (1-gamma)*np.exp(-((x-x0)/sigma)**2/2) + gamma/(1 + ((x-x0)/sigma)**2), (0, 0.5, 1)),
    ("Pseudo-Voigt", lambda x, x0, sigma, alpha: (1 - alpha) * np.exp(-((x-x0)/sigma)**2/2) + alpha/(1 + ((x-x0)/sigma)**2), (0, 0.5, 1)),
    ("Cauchy", lambda x, x0, gamma: gamma / (np.pi * (gamma**2 + (x-x0)**2)), (0, 1)),
    ("Student's t", lambda x, x0, sigma, nu: (1/(sigma*np.sqrt(np.pi*nu/2))) * (1 + ((x-x0)/sigma)**2/nu)**(-(nu+1)/2), (0, 1, 0.5)),
    ("Pearson", lambda x, x0, sigma, skew, kurtosis: (1/(sigma*np.sqrt(2*np.pi))) * ((1 + ((x-x0)/sigma)**2)**(-(skew+1)/(2*kurtosis))), (0, 0.5, 1, 1)),
    ("Sigmoid", lambda x, x0, k: 1/(1 + np.exp(-k*(x-x0))), (0, 2)),
    ("Squared cosine", lambda x, x0, c: np.where(np.abs(x-x0) <= c, 0.5*(1 + np.cos(np.pi*(x-x0)/c)), 0), (0, 2))
    ]


# Generate some x and y data for each function
# x = np.linspace(-15.5, 16.5, 1000)
y = [f(x, *parameters) for _, f, parameters in functions]

# Create the Bokeh figures for each function
figures = [figure(title=name) for name, f, parameters in functions]
for i, (name, f, parameters) in enumerate(functions):
    figures[i].line(x, y[i])

# Create a grid of the figures
grid = gridplot(children=figures, ncols = 3, merge_tools=False, width = 450, height = 320)
show(grid)


# -



# ## Fit raw data to base function 

# ### Fit to supergaussian 

# +
#| column: screen
from scipy.optimize import curve_fit

# Define the Gaussian function
def gaussian(x, mu, sigma, n, A):
    return A*np.exp(-abs(((x-mu)/sigma))**n)

# Set initial guess values for the parameters
initial_guess = [0, 1.5, 3, np.max(y_base)]

# # Fit the data to the Gaussian function using curve_fit
params, _ = curve_fit(gaussian, x_base, y_base, p0=initial_guess)

# # Extract the optimized parameter values
mu, sigma, n, A = params

# y = gaussian(x_base, 0, 1.3, 3.4, 35000)
# y = gaussian(x_base, mu, sigma, n, A)

kernel = np.ones(2000)
# y_convolved = np.convolve(y, kernel, mode='same')
# y_convolved = y_convolved/2000
plots_a = []
for sigma in np.arange(1.85, 2.05, 0.05):
    for n in np.arange(3.4, 4.0, 0.1):
        fit_plot = figure(x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
                    width = 300, height = 300)
        # print('sigma:',sigma)
        # print('n: ',n)
        y = gaussian(x_base, 0, sigma, n, 35000)
        corr_coef = np.corrcoef(y, y_base)[0,1]
        fit_plot.line(x_base, y_base, line_width = 5, color = '#9DD9C5')
        fit_plot.line(x_base, y, line_width = 5)
        fit_plot.title = f'sig: {sigma:.2f}; n: {n:.2f}; corr: {corr_coef:.4f}'
        # # fit_plot.line(x_base, y_convolved, line_width = 5, legend_label = 'convolved Gaussian', color = '#9D6C97')
        # fit_plot.x_range = Range1d(0, 6)
        # fit_plot.y_range = Range1d(-1000, 15000)
        fit_plot = plot_format(fit_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
        plots_a.append(fit_plot)


# print(params)
# show(fit_plot)
grid_fit = gridplot(children = plots_a, ncols = 7, merge_tools=False)
show(grid_fit)


# +
from scipy.optimize import curve_fit

# Define base function using xinterp and yinterp
def base_function(x, *params):
    return np.interp(x, x_base, y_base)

# Define function to fit to dataset1
def fit_function(x, a, b, c):
    return a * base_function(x, b, c)

# Extract xaxis and yaxis from dataset1
rough_df = pd.read_excel('data/rough_samples.xlsx')
xdata = rough_df['xaxis']
ydata = rough_df['pt2e']

# Use curve_fit to find parameters that minimize the difference between the data and the model
params, _ = curve_fit(fit_function, xdata, ydata)

# Evaluate the fitted function using the optimal parameters
yfit = fit_function(x_base, *params)

fit_plot = figure(title='Fit plot', x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 700, height = 500)

fit_plot.line(x_base, y_base, line_width = 5, color = '#9DD9C5')
# fit_plot.line(x_base, yfit, line_width = 4, color = '#9D6C97', line_dash="dotdash")
fit_plot.line(xdata, ydata, line_width = 5, color = '#9DC3E6')
fit_plot.circle(xdata, ydata, size = 8, color = '#9DC3E6', fill_color='#2F528F')
fit_plot = plot_format(fit_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")

show(fit_plot)


# +
def fit_function(x, a):
    return a * base_function(x)

from scipy.optimize import curve_fit

xdata = rough_df['xaxis']
ydata = rough_df['pt2e']

params, _ = curve_fit(fit_function, xdata, ydata)

yfit = fit_function(x_base, *params)

fit_plot = figure(title='Fit plot', x_axis_label='sampling point', y_axis_label='intensity', tooltips = [("index", "$index"),("(x,y)", "($x, $y)")],
            width = 700, height = 500)

fit_plot.line(x_base, y_base, line_width = 4, color = '#9DD9C5', legend = 'Base function', line_dash=[10, 5])
fit_plot.line(x_base, yfit, line_width = 4, color = '#9D6C97', legend = 'pt2 fitted function', line_dash="dotdash")
fit_plot.line(xdata, ydata, line_width = 5, color = '#9DC3E6', legend_label = 'pt2 line')
fit_plot.circle(xdata, ydata, size = 10, color = '#9DC3E6', fill_color='#2F528F', legend_label = 'pt2 points')
fit_plot = plot_format(fit_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
show(fit_plot)
# -

#

# ## Wafer 2 with rough areas 

# ### Calcute Aq, M and I values

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

# ### Non interpolated Plot

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

#

# +
#| echo: false
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

# +
#| echo: false
# zvalscut_hi = 2.04
# zvalscut_low = 1.96
# zvals = Aq
# zvalsnew = np.copy(zvals)
# zvalsnew[zvalsnew > zvalscut_hi] = zvalscut_hi
# zvalsnew[zvalsnew < zvalscut_low] = zvalscut_low

# matindx = []
# xaxis = np.arange(-55, 55.5, 0.5) * 1e-3
# yaxis = np.arange(-55, 55.5, 0.5) * 1e-3
# regmatAq = np.zeros((len(xaxis), len(yaxis)))
# nummat = np.zeros((len(xaxis), len(yaxis)))

# for xindx in range(len(xaxis)):
#     for yindx in range(len(yaxis)):
#         d = np.column_stack((xaxis[xindx] - xvals, yaxis[yindx] - yvals))
#         e = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
#         dindx = np.where(e < 0.0008)[0]
#         if len(dindx)>0:
#             regmatAq[xindx, yindx] = np.mean(zvalsnew[dindx])
#         else:
#             regmatAq[xindx, yindx] = np.nan

# regmatAq[0:2, 0] = [zvalscut_hi, zvalscut_low]

# +
#| echo: false
# # Create x and y coordinate arrays
# x, y = np.meshgrid(xaxis * 1e-3, yaxis * 1e-3)
# z = regmatAq
# source = pd.DataFrame({'x': x.ravel(),
#                      'y': y.ravel(),
#                      'z': z.ravel()})

# wafer_plot_interpolated = alt.Chart(source).mark_rect().encode(
#     x=alt.X('x:O', axis=None),
#     y=alt.Y('y:O', axis=None),
#     color=alt.Color('z:Q', scale=alt.Scale(scheme='turbo'))
# ).properties(width=400, height=400).interactive()

# wafer_plot_interpolated

# +
#| echo: false
# zvals = M
# df = pd.DataFrame({
#     'index':index,
#     'x': xvals*1000,
#     'y': yvals*1000,
#     'z': zvals,   
# })

# alt.Chart(df).mark_circle().encode(
#     x='x:Q', y='y:Q',
#     color= alt.Color('z:Q', scale=alt.Scale(scheme='Viridis'),
    
#     ),
#     tooltip=['x', 'y', 'z', 'index']
# ).properties(height=400, width=400).interactive()


# +
#| echo: false
# zvals = M
# zvalscut=0.2
# zvalsnew= np.array(zvals)
# zvalsnew[np.abs(zvalsnew) > zvalscut] = zvalscut

# df = pd.DataFrame({
#     'index':index,
#     'x': xvals*1000,
#     'y': yvals*1000,
#     'z': zvalsnew,   
# })

# alt.Chart(df).mark_circle().encode(
#     x='x:Q', y='y:Q',
#     color= alt.Color('z:Q', scale=alt.Scale(scheme='Turbo'),
    
#     ),
#     tooltip=['x', 'y', 'z', 'index']
# ).properties(height=400, width=400).interactive()


# +



