import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Range1d, Span
import numpy as np
import pandas as pd
from bokeh.palettes import Set3
# Create a Streamlit app
st.set_page_config(page_title="Super-Gaussian Equation Plotter", layout="wide")


def plot_format(plot, xlabel, ylabel, location, size, titlesize, labelsize):
    # x axis format
    plot.xaxis.axis_label = xlabel
    plot.xaxis.axis_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_style = "bold"
    plot.xaxis.axis_label_text_font_size = size
    plot.xaxis.major_label_text_font_size = size
    plot.xgrid.grid_line_color = '#2D3135'
    
    # y axis format
    plot.yaxis.axis_label = ylabel
    plot.yaxis.axis_label_text_font_style = 'bold'
    plot.yaxis.major_label_text_font_style = "bold"
    plot.yaxis.axis_label_text_font_size = size
    plot.yaxis.major_label_text_font_size = size
    plot.ygrid.grid_line_color = '#2D3135'

    # Legend format
    plot.legend.location = location
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = labelsize
    plot.legend.label_text_font_style = 'bold'
    plot.legend.border_line_width = 3
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"
    # plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.0

    # Title format
    plot.title.text_font_size = titlesize
    plot.background_fill_color = "#0E1117"
    plot.border_fill_color = "#0E1117"
    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "15pt"

    plot.legend.click_policy="hide"
    return plot



# 1. Define the background functions
functions = [
    ("Gaussian", lambda x, x0, sigma: np.exp(-((x-x0)/sigma)**2/2), (0.0, 1.3, 20000.0), (r'$x_0$ gaussian', r'$\sigma$ gaussian', 'amp_gaussian')),
    ("Lorentzian", lambda x, x0, gamma: 1/(1 + ((x-x0)/gamma)**2), (0.0, 1.0, 20000.0), (r'$x_0$ lorentzian', r'$\gamma$ lorentzian', 'amp_lorenzian')),
    ("Pseudo-Voigt", lambda x, x0, sigma, gamma: (1 - gamma) * np.exp(-((x-x0)/sigma)**2/2) + gamma/(1 + ((x-x0)/sigma)**2), (0.0, 0.5, 1.0, 20000.0), (r'$x_0$ voigt', r'$\sigma$ voigt', r'$\gamma$ voigt', 'amp_voigt')),
    ("Squared cosine", lambda x, x0, c: np.where(np.abs(x-x0) <= c, 0.5*(1 + np.cos(np.pi*(x-x0)/c)), 0), (0.0, 2.0, 20000.0), (r'$x_0$ cosine', 'c', 'amp_cosine'))
]

equations = [
    r"$\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$",
    r"$\frac{1}{1+\left(\frac{x-x_0}{\gamma}\right)^2}$",
    r"$(1-\gamma)\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right) + \frac{\gamma}{1+\left(\frac{x-x_0}{\sigma}\right)^2}$",
    r"$0.5*(1 + \cos(\pi\frac{(x-x_0)}{c}))$"
]

# 2. Get the base function xaxis and yaxis
st.sidebar.title("Function parameters")
base_bool = st.sidebar.checkbox("Plot base function", True)
background_bool = st.sidebar.checkbox("Plot background function", True)
exp_bool  = st.sidebar.checkbox("Plot experimental data", True)
add_bool = st.sidebar.checkbox("Add base and background functions ", True)
ylims = st.sidebar.slider("Select y axis range", -5000, 50000, (-5000, 45000), 1000)
xlims = st.sidebar.slider("Select x axis range", -20, 20, (-16, 16), 1)
base_shift = st.sidebar.number_input("Base function shift (degrees)", -6.0, 6.0, 0.0, 0.05)
base_amp = st.sidebar.slider("Base function amplitude", 0.0, 1.0, 0.35, 0.01)

base_function = pd.read_csv('data/base_funtion_interpolated.csv')
x_base = base_function['x_base'].copy().values
y_base = base_amp*base_function['y_base'].copy().values
x_background = base_function["x_base"].copy().values

# 3. Select experimental data
st.sidebar.title('Experimental data')
rough_df = pd.read_excel('data/rough_samples.xlsx')
columns = rough_df.columns
exp_data = st.sidebar.multiselect('Select experimental data', columns[1:], ['pt2d'])
color_palette = Set3[10]

# 3. Iterate over the lambda functions to create a slider per parameter
# figures = [figure(title=function[0], width = 650, height = 400) for function in functions] 
figures = [] 
for j, (name, f, params_nums, params_names) in enumerate(functions):
    # a. Add eq. in latex format and boolean to plot
    st.sidebar.title(name)
    st.sidebar.markdown(equations[j])
    plot_bool = st.sidebar.checkbox(f'Plot {name}', True)
    
    # b. Iterate over each parameter to create a slider
    values = []
    for i, param in enumerate(params_names):
        if 'amp' in param:
            value = st.sidebar.slider(param, 0, 45000, 20000, 1000)
        else:
            value = st.sidebar.slider(param, -5.0, 5.0, params_nums[i], 0.1)
        values.append(value)

    # c. Evaluate function
    # TODO: Shift y along x axis as well -@carlosreyes at 2/23/2023, 11:47:44 AM
    if plot_bool:
        p = figure(title = name, width=650, height=400)

        # c1. Shift base function axis
        x_base += base_shift # base_shift value from slider
        y_base = base_amp*y_base
        x_background += base_shift
        
        # c2. Calculate function
        values[0] = base_shift 
        y_background = values[-1]*f(x_background, *values[0:-1])
        y_sub = y_base + y_background
            
        # d. Plots
        # base function plot
        if base_bool:
            p.line(x_base, y_base, line_width = 5, color = '#9D6C97', legend_label = 'base_function')
            
        # background function plot
        if background_bool:
            p.line(x_background, y_background, line_width = 5, color = '#9DD9C5', legend_label = 'background_function')
        
        # rough data plot
        if exp_bool:
            for k, col in enumerate(exp_data):
                p.line(rough_df['xaxis'], rough_df[col], legend_label = col, line_width = 5, color=color_palette[k+1])
                p.circle(rough_df['xaxis'], rough_df[col], legend_label = col, size = 7)

        # base - background
        if add_bool:
            p.line(x_base, y_sub, line_width = 5, legend_label = 'Combined functions', color = '#A6DDFF', alpha = 1.0)
        
        p.x_range = Range1d(xlims[0], xlims[1])
        p.y_range = Range1d(ylims[0], ylims[1])
        p.xaxis.ticker.desired_num_ticks = 20
        p.yaxis.ticker.desired_num_ticks = 10
        vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
        p.add_layout(vline)
        p = plot_format(p, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
        figures.append(p)

    
grid = gridplot(children=figures, ncols = 2, merge_tools=False)
st.bokeh_chart(grid)


