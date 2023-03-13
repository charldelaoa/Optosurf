# ---
# title: "Base and background functions"
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
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.9.1 64-bit ('webapss')
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
from IPython.display import display
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
# -

# ## Base function

# An [interleaving algorithm](b_interleave.ipynb) was previously defined in order to increase the density of sampling points limited by the 32 pixels of the line detector. The algorithm consisted of performing a window integration (32 windows) over multiple optical fields with different $\mu$ parameters, which was equivalent to lateral shifts. Finally, an interpolation was done over the interleaved data, hence obtaining what was defined as the <span style="color:#EEDA89">base function</span>.
#
# This  <span style="color:#EEDA89">base function</span> represents a 'smooth surface' that will be then compared with rougher samples. In order to obtain an <span style="color:#EEDA89">experimental base function</span>, data was collected as illustrated in @fig-1 A. A smooth silicon wafer was rotated at different angles and then the 32 raw sampling points of the line detector were acquired. The collected angles were from -1.0 to 1.0 degrees.
#
# The following step is to sort the data points and apply the interleaving algorithm, with these, an <span style="color:#EEDA89">experimental base function</span> is obtained as illustrated in @fig-1 B.
#

# ::: {#fig-1}
#
# ![](images/c/Fig_1.png)
#
# A. Experimental data; B. Interleaved experimental base function
# :::

# ### Experimental base function data collection

# The following code shows the acquired experimental data:

# +
#| column: page
new_colors = []
for i in range(42):
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')

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
            width = 200, height = 150)
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

# The <span style="color:#EEDA89">experimental base function</span> is then obtained by interleaving the acquired experimental data. Note that some of the acquired points were very close to one another, hence they were creating a 'zig-zag' shape in the interpolation. In order to remove this, the data was made smoother by averaging values very close to one another. The code to obtain the <span style="color:#EEDA89">experimental base function</span> is now shown:

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

# b. Plot points
for (plot, df, legend, color) in [(interleaved_plot, base_function_df, 'Non-smooth base function', '#9DC3E6'), (smooth_plot, smooth_df, 'Smooth base function', '#9D6C97')]:
    # individual points
    plot.circle(df.xaxis, df.yaxis, color=df.colors, size=6)

    # smooth curve
    plot.line(df['xaxis'], df['yaxis'], line_width=4, legend_label=legend, color=color)

    # format
    plot.xaxis.ticker.desired_num_ticks = 15
    plot.y_range = Range1d(0, 45000)
    plot = plot_format(plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")

# d. Interpolation
x_base = np.arange(-15.5, 15.5001, 0.001).round(3)
pchip = PchipInterpolator(smooth_df['xaxis'], smooth_df['yaxis'])
y_base = pchip(x_base)

interpolated_plot.line(x=smooth_df['xaxis'], y=smooth_df['yaxis'], line_width = 5, legend_label = 'Smooth base function', color = '#9D6C97')
interpolated_plot.line(x_base, y_base, line_width = 5, color = '#9DD9C5', legend_label = 'Interpolated base function')
interpolated_plot.xaxis.ticker.desired_num_ticks = 15
interpolated_plot.y_range = Range1d(-1000, 45000)

interpolated_plot = plot_format(interpolated_plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
base_function_grid = gridplot(children=[interleaved_plot, smooth_plot, interpolated_plot], ncols=3, merge_tools=False, width=420, height=380)
show(base_function_grid)
# -

# ### Experimental rough data
# From the experimental data it was observed that rough samples modify the amplitude and tails of the base function, the data is now shown:

# +
#| column: page
from bokeh.palettes import Set3
# 1. Import data
rough_df = pd.read_excel('data/rough_samples.xlsx')
source_rough = ColumnDataSource(rough_df)

# # 2. Create plot
rough_plots = []
color_palette = Set3[len(rough_df.columns[1:])+2]

# a. iterate over the columns and add a line for each one
for i, col in enumerate(rough_df.columns[1:]):
    rough_plot = figure(title = str(col), x_axis_label='xaxis', y_axis_label='yaxis', width = 470, height = 420, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    # Base function points
    
    rough_plot.circle(x=base_function_df.xaxis, y=base_function_df.yaxis, color = base_function_df.colors, legend_label = 'Base function non-smooth', size = 6)
    rough_plot.circle(x=smooth_df.xaxis, y=smooth_df.yaxis, color = smooth_df.colors, legend_label = 'Base function smooth', size = 6)
    # rough_plot.line(x=smooth_df.xaxis, y=smooth_df.yaxis, legend_label = 'Base function', line_width=4, color='#D17B8F')

    # Experimental data
    rough_plot.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4)
    rough_plot.triangle('xaxis', col, source=source_rough, fill_color= color_palette[1], size=10, legend_label = f"{col} points")
    
    # Plot format
    rough_plot.y_range = Range1d(-5000, 50000)
    rough_plot = plot_format(rough_plot, "Degrees", "Intensity", "top_left", "10pt", "8pt", "8pt")
    rough_plots.append(rough_plot)

grid_rough = gridplot(children = rough_plots, ncols = 3, merge_tools=False, width = 370, height = 340)
show(grid_rough)
# -

# ## Background function

# The <span style="color:#EEDA89">experimental base function</span> is the reference numerical function obtained from a smooth wafer at different angles. In practice the wafer roughness is to be measured for wafers with different roughness. Mathematically, this represents a change in amplitude and additional tails in the <span style="color:#EEDA89">base function</span>  function. This is illustrated in @fig-2. Hence, different parameters have to be found in order to approximate the base function to the real rough experimental data. 

# ::: {#fig-2}
#
# <div style="text-align:center"><img src="images/c/Fig_2.png" alt="Base function" width="550"/></div>
#
# The base function is to be used a reference function modified by n parameters in order to approximate to the roughness data.
# :::

# In order to go from a smooth base function to rough data, a <span style="color:#9DD9C5">background function</span> is added in order to modify the amplitude and tails. This is illustrated in @fig-3. The 'smooth' base function (a) is modified by adding a background function (b), e.g., a Gaussian or Lorentzian function with their corresponding amplitude, $\sigma$ and $\gamma$ parameters. The output of the addition will be a <span style="color:#B77E61">modified function</span> (c). The final step is to <span style="color:#B77E61">downsample the modified</span> (d) function and compare it with the experimental rough data by using an error function.

# ::: {#fig-3}
#
# <div style="text-align:center"><img src="images/c/Fig_3.png" alt="modified function" width="600"/></div>
#
# A. Base function; B. Backgroun function (Gaussian/Lorentzian); C. Modified function; D. Experimental data
# :::

# ## Modified function

# Once the base function has been numerically defined, then the background function can be added in order to obtained a modified function that approximates the rough data. The proposed background functions are:
#
# 1. **Gaussian**: $A\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$, with parameters $x_{o}$, $\sigma$ and $A$. These parameters were tuned to $x_{o}=0$, $\sigma=1.9$, $A=3500$
#
# 2. **Lorentzian**: $A\frac{1}{1+\left(\frac{x-x_0}{\gamma}\right)^2}$, with parameters $x_{o}$, $\gamma$ and $A$. These parameters were tuned to $x_{o}=0$ and $\gamma=2.1$, $A=2500$
#
# The base function (purple) multiplied by a factor of 0.8, after this both background functions (green curve) are added in order to modify the amplitude and tails. The resulting modified function (blue) is then downsampled(brown-dashed-triangles) and compared with real experimental rough data (yellow). Notice that if you click in the plot label you can hide the data.

# +
#| column: page
from bokeh.palettes import Set3
color_palette = Set3[10]

# 1. Define functions
functions = [
    ("Gaussian", lambda x, x0, sigma: np.exp(-((x-x0)/sigma)**2/2), (0.0, 1.9, 3500, 0.8), (r'$x_0$ gaussian', r'$\sigma$ gaussian', 'amp_gaussian', 'base function amplitude 1')),
    ("Lorentzian", lambda x, x0, gamma: 1/(1 + ((x-x0)/gamma)**2), (0.0, 2.1, 2500, 0.8), (r'$x_0$ lorentzian', r'$\gamma$ lorentzian', 'amp_lorenzian', 'base function amplitude 2'))]
labels = ["1. ", "2. "]

equations = [
    r"$\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$",
    r"$\frac{1}{1+\left(\frac{x-x_0}{\gamma}\right)^2}$"]

# 2. Get base function
base_function = pd.read_csv('data/base_funtion_interpolated.csv')
x_base = base_function['x_base'].copy().values.round(3)
y_base = base_function['y_base'].copy().values
x_background = base_function["x_base"].copy().values

# 3. Get rough data
rough_df = pd.read_excel('data/rough_samples.xlsx')
x_rough = rough_df["xaxis"].copy().values
# y_rough = ['ann1', 'pt2', 'pt2b', 'pt2c', 'pt2d', 'pt2e']
y_rough = ['pt2d']
columns = list(rough_df.columns)
figures = [] 
for j, (name, f, params_nums, params_names) in enumerate(functions):
    p = figure(title = f"{labels[j]} {name}", width=750, height=450)
    # 3. Shift base function axis
    x_base += params_nums[0] 
    y_base = params_nums[-1]*base_function['y_base'].copy().values
    x_background += params_nums[0]
     
    # 4. Calculate background function
    y_background = params_nums[-2]*f(x_background, *params_nums[0:-2])
    y_final = y_base + y_background 

    # 5. Plots
    # 5.1 base function plot
    p.line(x_base, y_base, line_width = 5, color = '#9D6C97', legend_label = 'base_function')
    vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
    p.add_layout(vline) 
    
    # 5.2 Background function plot
    p.line(x_background, y_background, line_width = 5, color = '#9DD9C5', legend_label = 'background_function')
    
    # 5.3 Modified function
    indices = np.where(np.isin(x_base, x_rough+params_nums[0]))[0]
    y_final_points = y_final[indices]
    p.line(x_base, y_final, line_width = 5, legend_label = 'Base + background functions', color = '#A6DDFF', alpha = 1.0)
    
    # 5.4 Plot format
    p.xaxis.ticker.desired_num_ticks = 10
    p.yaxis.ticker.desired_num_ticks = 10  
    p.y_range = Range1d(-5000, 45000)  
    p = plot_format(p, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
    figures.append(p)

    # 5.5 Rough data plot
    corr_coef = np.corrcoef(y_final_points, rough_df['pt2d'])[0,1]
    
    p2 = figure(title = f"{name} downsampling; correlation coefficient: {corr_coef:.4f}", width=750, height=450)
    p2.line(x_base, y_final, line_width = 5, legend_label = 'Base + background functions', color = '#A6DDFF', alpha = 1.0)
    k = 0
    for col in columns[1:]:
        if col in y_rough:
            p2.line(x_rough, rough_df[col], legend_label = col, line_width = 5, color=color_palette[k+1])
            p2.circle(x_rough, rough_df[col], legend_label = col, size = 7, color='#5F9545')
            k+=1      
    
    # 5.6 Downsampled data
    p2.line(x_rough+params_nums[0], y_final_points, line_width=5, legend_label = 'Downsampling', color = '#98473E',  alpha = 0.7, line_dash='dashed')
    p2.triangle(x_rough+params_nums[0], y_final_points, size = 10, legend_label = 'Downsampling', color = '#DB8A74')

    # 5.7 plot format
    p2 = plot_format(p2, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
    p2.xaxis.ticker.desired_num_ticks = 10
    p2.yaxis.ticker.desired_num_ticks = 10  
    p2.y_range = Range1d(-5000, 50000)  
    figures.append(p2)

grid_modified = gridplot(children = figures, ncols = 2, merge_tools=False, width=420, height = 390)
show(grid_modified)
# -

# ## Minimization/optimization function 
#
# In order to obtain the optimized parameters to math the experimental function a minimization/optimization function was implemented. The algorithm consists of (@fig-4 ):
#
# 1. Give an initial guess for the background parameters.
# 2. Use the interleaved base function as a reference.
# 3. Shift the refence axis with respect to the initial guess of x0
# 4. Interpolate the base function to the new axis (32 points)
# 5. Calculate the background function
# 6. Obtain the modified function = base + background
# 7. Optimize least square difference with respect to the experimental data
# 8. Obtain the optimized parameters
#

# ::: {#fig-4}
#
# <div style="text-align:center"><img src="images/c/Fig_4.png" alt="Streamlit webapp" width="850"/></div>
#
# Streamlit webapp
# :::

# ### Gaussian function: $A_{1}\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$
#
# In order to obtain a modified function as close as possible to the experimental the parameters that modify the base and background function have to be optimized according to a minimization function. These parameters are:
#
# *`x0:` introduces a lateral shifth to both the base and background functions
#
# *`Abase:` modified the amplitude of the base function
#
# *`sigma:` sigma parameter of the background gaussian function
#
# *`A1:` amplitude of the background function

# +
#| column: screen
from bokeh.palettes import Set3
from scipy.optimize import minimize

# 1. Get base function points (330 points from -16.4 to 16.5)
x_base = smooth_df.xaxis.values
y_base = smooth_df.yaxis.values
# x_base = np.concatenate(([-50000], x_base, [50000]))
# y_base = np.concatenate(([0], y_base, [0]))

# 2. get rough data 
rough_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Data')
source_rough = ColumnDataSource(rough_df)
x_rough = rough_df['xaxis'].values.round(3)

# 3. Get initial guesses
guess_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Gaussian')
guess_df = guess_df.set_index('Variables')

# 4. Create df that will save optmized parameters
columns = ['ann1_opt', 'pt2_opt', 'pt2b_opt', 'pt2c_opt', 'pt2d_opt', 'pt2e_opt']
index = ['x0', 'Abase', 'sigma', 'Agaussian']
optimized_df = pd.DataFrame(columns=columns, index=index)

# 5. Define gaussian function
gaussian = lambda x, x0, sigma, A1: A1 * np.exp(-((x - x0) / sigma) ** 2 / 2)

# 6. Define cost function
pchip = PchipInterpolator(x_base, y_base)
def cost_function(params, y):
    x0, A0, sigma, A1 = params
    # Get new x axis
    x_new = x_rough + x0 
    # interpolate base function with respect to x_new (32 points)
    y_base_modified = A0*pchip(x_new) 
    # calculate background on original axis and with x0
    y_background = gaussian(x_new, x0, sigma, A1)
    # calculate modified function
    y_modified = y_base_modified + y_background
    # Compare directly with 32 points experimental data
    return np.sum((y_modified - y) ** 2)

# 7. Iterate over the experimental data
rough_plots = []
color_palette = Set3[len(rough_df.columns[1:])+2]
bounds = ((None, None), (0, None), (0, None), (0, None))
backgrounds = figure(title = 'Background functions with experimental data', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
backgroundsbg = figure(title = 'Background functions', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
downsamplesg = figure(title = 'Gaussian downsampled points', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
differenceg = figure(title = 'Gaussian Experimental vs Optimized differences', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])

for i, col in enumerate(rough_df.columns[1:]):
    # 8. Get initial guesses
    rough_plot = figure(title = str(col), width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    guess = [guess_df.loc[var][col] for var in ['x0', 'Abase', 'sigma', 'Agaussian']]
    # x0 = params[0]

    # 8. Call minimization function
    y_rough = rough_df[col].copy().values
    cost_fn = lambda p:cost_function(p, y_rough)
    result = minimize(cost_fn, guess)
    # result = minimize(cost_fn, params, bounds=bounds)
    optimized_parameters = result.x
    colu = col + '_opt'
    x0_opt, A0_opt, sigma_opt, A1_opt = optimized_parameters
    optimized_df.loc['x0'][colu] = x0_opt
    optimized_df.loc['Abase'][colu] = A0_opt
    optimized_df.loc['sigma'][colu] = sigma_opt
    optimized_df.loc['Agaussian'][colu] = A1_opt
    
    # 7. Calculate new optimized modified function
    x_new_opt = x_rough + x0_opt 
    # interpolate base function with respect to x_new (32 points)
    y_base_opt = A0_opt*pchip(x_new_opt) 
    # calculate background on original axis and with x0
    y_background_opt = gaussian(x_new_opt, x0_opt, sigma_opt, A1_opt)
    # calculate optmized function
    y_optimized = y_base_opt + y_background_opt

    vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
    rough_plot.add_layout(vline)

    # Plot optimize function lines
    rough_plot.line(x_rough, y_base_opt, legend_label = 'Base', line_width = 5, color='#F96F5D')
    rough_plot.line(x_rough, y_background_opt, legend_label = 'Bbackground', line_width = 5, color='#F9B5AC')
    rough_plot.line(x_rough, y_optimized, legend_label = 'Optimized function', line_width = 5, color='#987284')
    rough_plot.triangle(x_rough, y_optimized, legend_label = 'Optimized points', size = 8, color=color_palette[1])
    backgrounds.line(x_rough, y_background_opt, color = color_palette[i], line_width = 5 , legend_label = f"Background {col}")
    backgroundsbg.line(x_rough, y_background_opt, color = color_palette[i], line_width = 5 , legend_label = f"Background {col}")
    downsamplesg.line(x_rough, y_optimized, line_width=4, legend_label = f'Downsampling {col}', color = color_palette[i+1],  alpha = 0.9, line_dash='dashed')
    downsamplesg.triangle(x_rough, y_optimized, size = 13, legend_label = f'Downsampling {col}', color = color_palette[i+1])

    # Plot rough experimental data
    rough_plot.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4, line_dash = 'dashed')
    rough_plot.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = f"{col} points")
    backgrounds.line('xaxis', col, source=source_rough, color = color_palette[i], legend_label = str(col), line_width=4)
    backgrounds.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    downsamplesg.line('xaxis', col, source=source_rough, color = color_palette[i], legend_label = str(col), line_width=4)
    downsamplesg.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    
    # Plot format
    rough_plot.y_range = Range1d(-5000, 50000)
    rough_plot.xaxis.ticker.desired_num_ticks = 10
    rough_plot.yaxis.ticker.desired_num_ticks = 10
    rough_plot = plot_format(rough_plot, "Degrees", "Intensity", "top_left", "10pt", "8pt", "9pt")
    rough_plots.append(rough_plot)

    # Difference plot
    diff = y_rough - y_optimized
    differenceg.line(x=x_rough, y=diff, legend_label = col, color = color_palette[i], line_width=4)
    differenceg.circle(x=x_rough, y=diff, legend_label = col, fill_color= color_palette[i], size=7)

plots = [backgrounds, backgroundsbg, downsamplesg, differenceg]
for plot in plots:
    plot = plot_format(plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
    plot.y_range = Range1d(-5000, 50000)
    rough_plots.append(plot)
    plot.xaxis.ticker.desired_num_ticks = 10
    plot.yaxis.ticker.desired_num_ticks = 10
backgroundsbg.y_range = Range1d(-1000, 6000)
differenceg.y_range = Range1d(-3000, 3000)

grid_rough = gridplot(children = rough_plots, ncols = 3, merge_tools=False, width = 480, height = 430)
show(grid_rough)

merged_df = guess_df.join(optimized_df)\
    [['ann1', 'ann1_opt', 'pt2', 'pt2_opt', 'pt2b', 'pt2b_opt', 'pt2c', 'pt2c_opt', 'pt2d', 'pt2d_opt', 'pt2e', 'pt2e_opt']]
display(merged_df)
# -

# ### Lorentzian function:  $A_{2} \frac{1}{1+\left(\frac{x-x_0}{\gamma}\right)^2}$
#
# In order to obtain a modified function as close as possible to the experimental the parameters that modify the base and background function have to be optimized according to a minimization function. These parameters are:
#
# *`x0:` introduces a lateral shifth to both the base and background functions
#
# *`A0:` modified the amplitude of the base function
#
# *`gamma:` gamma parameter of the background gaussian function
#
# *`A2:` amplitude of the background function

# +
#| column: screen
from bokeh.palettes import Set3
from scipy.optimize import minimize

# 1. Get base function points (330 points from -16.4 to 16.5)
x_base = smooth_df.xaxis.values
y_base = smooth_df.yaxis.values

# 2. get rough data 
rough_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Data')
source_rough = ColumnDataSource(rough_df)
x_rough = rough_df['xaxis'].values.round(3)

# 3. Get initial guesses
guess_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Lorentzian')
guess_df = guess_df.set_index('Variables')

# 4. Create df that will save optmized parameters
columns = ['ann1_opt', 'pt2_opt', 'pt2b_opt', 'pt2c_opt', 'pt2d_opt', 'pt2e_opt']
index = ['x0', 'Abase', 'gamma', 'Alorentzian']
optimized_df = pd.DataFrame(columns=columns, index=index)

# 5. Define lorentzian function
lorentzian = lambda x, x0, gamma, A1: A1/(1 + ((x-x0)/gamma)**2)

# 6. Define cost function
pchip = PchipInterpolator(x_base, y_base)
def cost_function(params, y):
    x0, A0, gamma, A1 = params
    # Get new x axis
    x_new = x_rough + x0 
    # interpolate base function with respect to x_new (32 points)
    y_base_modified = A0*pchip(x_new) 
    # calculate background on original axis and with x0
    y_background = lorentzian(x_new, x0, gamma, A1)
    # calculate modified function
    y_modified = y_base_modified + y_background
    # Compare directly with 32 points experimental data
    return np.sum((y_modified - y) ** 2)

# 7. Iterate over the experimental data
rough_plots = []
color_palette = Set3[len(rough_df.columns[1:])+2]
bounds = ((None, None), (0, None), (0, None), (0, None))
backgrounds = figure(title = 'Background functions with experimental data', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
backgroundsbl = figure(title = 'Background functions', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
downsamples_l = figure(title = 'Lorentzian Downsampled points', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
differencel = figure(title = 'Lorentzian Experimental vs Optimized differences', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])

for i, col in enumerate(rough_df.columns[1:]):
    # 8. Get initial guesses
    rough_plot = figure(title = str(col), width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    guess = [guess_df.loc[var][col] for var in ['x0', 'Abase', 'gamma', 'Alorentzian']]
    
    # 9. Call minimization function
    y_rough = rough_df[col].copy().values
    cost_fn = lambda p:cost_function(p, y_rough)
    result = minimize(cost_fn, guess)
    optimized_parameters = result.x
    colu = col + '_opt'
    x0_opt, A0_opt, gamma_opt, A1_opt = optimized_parameters
    optimized_df.loc['x0'][colu] = x0_opt
    optimized_df.loc['Abase'][colu] = A0_opt
    optimized_df.loc['gamma'][colu] = gamma_opt
    optimized_df.loc['Alorentzian'][colu] = A1_opt
    
    # 10. Calculate optimized function
    # Calculate new optimized xaxis
    x_new_opt = x_rough + x0_opt 
    # interpolate base function with respect to x_new (32 points)
    y_base_opt = A0_opt*pchip(x_new_opt) 
    # calculate background on original axis and with x0
    y_background_opt = lorentzian(x_new_opt, x0_opt, gamma_opt, A1_opt)
    # calculate optmized function
    y_optimized = y_base_opt + y_background_opt

    # 11. Plot optimize function lines
    vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
    rough_plot.add_layout(vline)
    rough_plot.line(x_rough, y_base_opt, legend_label = 'Base', line_width = 5, color='#F96F5D')
    rough_plot.line(x_rough, y_background_opt, legend_label = 'Bbackground', line_width = 5, color='#F9B5AC')
    rough_plot.line(x_rough, y_optimized, legend_label = 'Optimized function', line_width = 5, color='#987284')
    rough_plot.triangle(x_rough, y_optimized, legend_label = 'Optimized points', size = 8, color=color_palette[1])
    backgrounds.line(x_rough, y_background_opt, color = color_palette[i], line_width = 5 , legend_label = f"Background {col}")
    backgroundsbl.line(x_rough, y_background_opt, color = color_palette[i], line_width = 5 , legend_label = f"Background {col}")
    downsamples_l.line(x_rough, y_optimized, line_width=4, legend_label = f'Downsampling {col}', color = color_palette[i+1],  alpha = 0.9, line_dash='dashed')
    downsamples_l.triangle(x_rough, y_optimized, size = 9, legend_label = f'Downsampling {col}', color = color_palette[i+1])

    # 12. Plot rough experimental data
    rough_plot.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4, line_dash = 'dashed')
    rough_plot.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = f"{col} points")
    backgrounds.line('xaxis', col, source=source_rough, color = color_palette[i], legend_label = str(col), line_width=4)
    backgrounds.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    downsamples_l.line('xaxis', col, source=source_rough, color = color_palette[i], legend_label = str(col), line_width=4)
    downsamples_l.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    
    # 13. Plot format
    rough_plot.y_range = Range1d(-5000, 50000)
    rough_plot.xaxis.ticker.desired_num_ticks = 10
    rough_plot.yaxis.ticker.desired_num_ticks = 10
    rough_plot = plot_format(rough_plot, "Degrees", "Intensity", "top_left", "10pt", "8pt", "9pt")
    rough_plots.append(rough_plot)

    # Difference plot
    diff = y_rough - y_optimized
    differencel.line(x=x_rough, y=diff, legend_label = col, color = color_palette[i], line_width=4)
    differencel.circle(x=x_rough, y=diff, legend_label = col, fill_color= color_palette[i], size=7)

plots = [backgrounds, backgroundsbl, downsamples_l, differencel]
for plot in plots:
    plot = plot_format(plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
    plot.y_range = Range1d(-5000, 50000)
    plot.xaxis.ticker.desired_num_ticks = 10
    plot.yaxis.ticker.desired_num_ticks = 10
    rough_plots.append(plot)
backgroundsbl.y_range = Range1d(-1000, 6000)
differencel.y_range = Range1d(-3000, 3000)

grid_rough = gridplot(children = rough_plots, ncols = 3, merge_tools=False, width = 480, height = 430)
show(grid_rough)

merged_df = guess_df.join(optimized_df)\
    [['ann1', 'ann1_opt', 'pt2', 'pt2_opt', 'pt2b', 'pt2b_opt', 'pt2c', 'pt2c_opt', 'pt2d', 'pt2d_opt', 'pt2e', 'pt2e_opt']]
display(merged_df)
# -

# ### Pseudo-voigt $A_{1}*(1-\gamma)\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right) + A2*\frac{\gamma}{1+\left(\frac{x-x_0}{\sigma}\right)^2}$

# +
#| column: screen
from bokeh.palettes import Set3
from scipy.optimize import minimize

# 1. Get base function points (330 points from -16.4 to 16.5)
x_base = smooth_df.xaxis.values
y_base = smooth_df.yaxis.values

# 2. get rough data 
rough_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Data')
source_rough = ColumnDataSource(rough_df)
x_rough = rough_df['xaxis'].values.round(3)

# 3. Get initial guesses
guess_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Voigt')
guess_df = guess_df.set_index('Variables')

# 4. Create df that will save optmized parameters
columns = ['ann1_opt', 'pt2_opt', 'pt2b_opt', 'pt2c_opt', 'pt2d_opt', 'pt2e_opt']
index = ['x0', 'Abase', 'sigma', 'gamma', 'A1', 'A2']
optimized_df = pd.DataFrame(columns=columns, index=index)

# 5. Define lorentzian function
voigt = lambda x, x0, sigma, gamma, A1, A2: A1*(1 - gamma) * np.exp(-((x-x0)/sigma)**2/2) + A2*gamma/(1 + ((x-x0)/sigma)**2)

# 6. Define cost function
pchip = PchipInterpolator(x_base, y_base)
def cost_function(params, y):
    x0, Abase, sigma, gamma, A1, A2 = params
    # Get new x axis
    x_new = x_rough + x0 
    # interpolate base function with respect to x_new (32 points)
    y_base_modified = Abase*pchip(x_new) 
    # calculate background on original axis and with x0
    y_background = voigt(x_new, x0, sigma, gamma, A1, A2)
    # calculate modified function
    y_modified = y_base_modified + y_background
    # Compare directly with 32 points experimental data
    return np.sum((y_modified - y) ** 2)

# 7. Iterate over the experimental data
rough_plots = []
color_palette = Set3[len(rough_df.columns[1:])+2]
bounds = ((None, None), (0, None), (0, None), (0, None))
backgrounds = figure(title = 'Background functions with experimental data', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
backgroundsbv = figure(title = 'Background functions', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
downsamples_v = figure(title = 'Voigt Downsampled points', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
differencev = figure(title = 'Voigt Experimental vs Optimized differences', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])

for i, col in enumerate(rough_df.columns[1:]):
    # 8. Get initial guesses
    rough_plot = figure(title = str(col), width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
    guess = [guess_df.loc[var][col] for var in ['x0', 'Abase', 'sigma', 'gamma', 'A1', 'A2']]
    
    # 9. Call minimization function
    y_rough = rough_df[col].copy().values
    cost_fn = lambda p:cost_function(p, y_rough)
    result = minimize(cost_fn, guess)
    optimized_parameters = result.x
    colu = col + '_opt'
    x0_opt, Abase_opt, sigma_opt, gamma_opt, A1_opt, A2_opt = optimized_parameters

    optimized_df.loc['x0'][colu] = x0_opt
    optimized_df.loc['Abase'][colu] = Abase_opt
    optimized_df.loc['sigma'][colu] = sigma_opt
    optimized_df.loc['gamma'][colu] = gamma_opt
    optimized_df.loc['A1'][colu] = A1_opt
    optimized_df.loc['A2'][colu] = A2_opt
    
    # 10. Calculate optimized function
    # Calculate new optimized xaxis
    x_new_opt = x_rough + x0_opt 
    # interpolate base function with respect to x_new (32 points)
    y_base_opt = Abase_opt*pchip(x_new_opt) 
    # calculate background on original axis and with x0
    y_background_opt = voigt(x_new_opt, x0_opt, sigma_opt, gamma_opt, A1_opt, A2_opt)
    # calculate optmized function
    y_optimized = y_base_opt + y_background_opt

    # 11. Plot optimize function lines
    vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
    rough_plot.add_layout(vline)
    rough_plot.line(x_rough, y_base_opt, legend_label = 'Base', line_width = 5, color='#F96F5D')
    rough_plot.line(x_rough, y_background_opt, legend_label = 'Bbackground', line_width = 5, color='#F9B5AC')
    rough_plot.line(x_rough, y_optimized, legend_label = 'Optimized function', line_width = 5, color='#987284')
    rough_plot.triangle(x_rough, y_optimized, legend_label = 'Optimized points', size = 8, color=color_palette[1])

    backgrounds.line(x_rough, y_background_opt, color = color_palette[i], line_width = 5 , legend_label = f"Background {col}")
    backgroundsbv.line(x_rough, y_background_opt, color = color_palette[i], line_width = 5 , legend_label = f"Background {col}")

    downsamples_v.line(x_rough, y_optimized, line_width=4, legend_label = f'Downsampling {col}', color = color_palette[i+1],  alpha = 0.9, line_dash='dashed')
    downsamples_v.triangle(x_rough, y_optimized, size = 9, legend_label = f'Downsampling {col}', color = color_palette[i+1])

    # 12. Plot rough experimental data
    rough_plot.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4, line_dash = 'dashed')
    rough_plot.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = f"{col} points")

    backgrounds.line('xaxis', col, source=source_rough, color = color_palette[i], legend_label = str(col), line_width=4)
    backgrounds.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    
    downsamples_v.line('xaxis', col, source=source_rough, color = color_palette[i], legend_label = str(col), line_width=4)
    downsamples_v.circle('xaxis', col, source=source_rough, fill_color= color_palette[i], size=7, legend_label = str(col))
    
    # 13. Plot format
    rough_plot.y_range = Range1d(-5000, 50000)
    rough_plot.xaxis.ticker.desired_num_ticks = 10
    rough_plot.yaxis.ticker.desired_num_ticks = 10
    rough_plot = plot_format(rough_plot, "Degrees", "Intensity", "top_left", "10pt", "8pt", "9pt")
    rough_plots.append(rough_plot)

    # Difference plot
    diff = y_rough - y_optimized
    differencev.line(x=x_rough, y=diff, legend_label = col, color = color_palette[i], line_width=4)
    differencev.circle(x=x_rough, y=diff, legend_label = col, fill_color= color_palette[i], size=7)

plots = [backgrounds, backgroundsbv, downsamples_v, differencev]
for plot in plots:
    plot = plot_format(plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
    plot.y_range = Range1d(-5000, 50000)
    plot.xaxis.ticker.desired_num_ticks = 10
    plot.yaxis.ticker.desired_num_ticks = 10
    rough_plots.append(plot)
backgroundsbv.y_range = Range1d(-1000, 6000)
differencev.y_range = Range1d(-3000, 3000)

grid_rough = gridplot(children = rough_plots, ncols = 3, merge_tools=False, width = 480, height = 430)
show(grid_rough)

merged_df = guess_df.join(optimized_df)\
    [['ann1', 'ann1_opt', 'pt2', 'pt2_opt', 'pt2b', 'pt2b_opt', 'pt2c', 'pt2c_opt', 'pt2d', 'pt2d_opt', 'pt2e', 'pt2e_opt']]
display(merged_df)
# -

# ### Results summary 
#

#| column: screen
grid_samples = gridplot(children = [downsamplesg, downsamples_l, downsamples_v, differenceg, differencel, differencev, backgroundsbg, backgroundsbl, backgroundsbv], ncols = 3, merge_tools=False, width = 570, height = 450)
show(grid_samples)

# Plot background functions from all fits

# ## Conclusions

# 1. An experimental base function was obtained from a smooth wafer at different angles.
# 2. Three background functions were proposed in order to obtain a modified function that matches the experimental rough data, a Gaussian, a Lorentzian and Voigt function with their corresponding parameters.
# 3. These parameters were optimized/minimized in order to match the experimental data.
# 4. It was observed that indeed the base function plus the background function is equivalent to 'adding' roughness to a smooth wafer. This was observed in an amplitude and tails change.

# ## Simulation WebApp
# A web application including all the previous functions can be access here: [Streamlit app](https://charldelaoa-optosurf-streamlitc-experimental-data-st-jg9o2q.streamlit.app)
#
# ::: {#fig-5}
#
# ![](images/c/Fig_5.png)
#
# Streamlit webapp
# :::

# +
# take my 330 points (pchip interpolation to a new axis) 
# introduce a tilt step as I'm shifting axis, calculate histogram and see the variation of the Aq number
# E.g ann1 is rougher than pt2, we could measure negative 'roughness' on pt2
# Aq numbers might be the other way around
# -

#
