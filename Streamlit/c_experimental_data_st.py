import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Range1d, Span
import numpy as np
import pandas as pd
from bokeh.palettes import Set3
from pathlib import Path
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
    ("Gaussian", lambda x, x0, sigma: np.exp(-((x-x0)/sigma)**2/2), (0.0, 1.9, 3500, 0.8), (r'$x_0$ gaussian', r'$\sigma$ gaussian', 'amp_gaussian', 'base function amplitude 1')),
    ("Lorentzian", lambda x, x0, gamma: 1/(1 + ((x-x0)/gamma)**2), (0.0, 2.1, 2500, 0.82), (r'$x_0$ lorentzian', r'$\gamma$ lorentzian', 'amp_lorenzian', 'base function amplitude 2')),
    ("Pseudo-Voigt", lambda x, x0, sigma, gamma: (1 - gamma) * np.exp(-((x-x0)/sigma)**2/2) + gamma/(1 + ((x-x0)/sigma)**2), (0.0, 0.2, 1.5, 17750, 0.8), (r'$x_0$ voigt', r'$\sigma$ voigt', r'$\gamma$ voigt', 'amp_voigt', 'base function amplitude 3')),
    ("Squared cosine", lambda x, x0, c: np.where(np.abs(x-x0) <= c, 0.5*(1 + np.cos(np.pi*(x-x0)/c)), 0), (0.0, 2.0, 20000, 0.5), (r'$x_0$ cosine', 'c', 'amp_cosine', 'base function amplitude 4'))
]

equations = [
    r"$\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$",
    r"$\frac{1}{1+\left(\frac{x-x_0}{\gamma}\right)^2}$",
    r"$(1-\gamma)\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right) + \frac{\gamma}{1+\left(\frac{x-x_0}{\sigma}\right)^2}$",
    r"$0.5*(1 + \cos(\pi\frac{(x-x_0)}{c}))$"
]

# 2. Define sliders
st.sidebar.title("Function parameters")
base_bool = st.sidebar.checkbox("Plot base function", True)
background_bool = st.sidebar.checkbox("Plot background function", True)
exp_bool  = st.sidebar.checkbox("Plot experimental data", True)
add_bool = st.sidebar.checkbox("Add base and background functions ", True)
downsampling_bool = st.sidebar.checkbox("Add downsampled points ", True)
ylims = st.sidebar.slider("Select y axis range", -5000, 50000, (-5000, 45000), 1000)
xlims = st.sidebar.slider("Select x axis range", -20, 20, (-16, 16), 1)

# 3. Get base function
path = Path(__file__).parents[0] / 'base_funtion_interpolated.csv' 
st.write(path)
base_function = pd.read_csv(path)
x_base = base_function['x_base'].copy().values.round(3)
y_base = base_function['y_base'].copy().values
x_background = base_function["x_base"].copy().values

# 4. Select experimental data
st.sidebar.title('Experimental data')
path = Path(__file__).parents[0] / 'rough_samples.xlsx'
rough_df = pd.read_excel(path)
x_rough = rough_df["xaxis"].copy().values
columns = rough_df.columns
exp_data = st.sidebar.multiselect('Select experimental data', columns[1:], ['pt2d'])
color_palette = Set3[10]

# 5. Iterate over the lambda functions to create a slider per parameter
figures = [] 
for j, (name, f, params_nums, params_names) in enumerate(functions):
    # 6. Add eq. in latex format and boolean to plot
    st.title(f"{name}: {equations[j]}")
    # st.markdown(equations[j])
    plot_bool = st.checkbox(f'Plot {name}', True)
    columns = st.columns([1,2])
    # 7. Iterate over each parameter to create a slider
    values = []
    for i, param in enumerate(params_names):
        with columns[0]:
            if 'amp_' in param:
                value = st.slider(param, 0, 45000, params_nums[i], 250)
            elif 'x' in param:
                value = st.number_input(param, -6.0, 6.0, params_nums[i], 0.05)
            elif 'base' in param:
                base_amp = st.slider(param, 0.0, 1.0, params_nums[i], 0.01)
            else:
                value = st.slider(param, -5.0, 5.0, params_nums[i], 0.1)
            values.append(value)

    if plot_bool:
        p = figure(title = name, width=770, height=530) 
        # 8. Shift base function axis
        x_base += values[0] # base_shift value from slider
        y_base = base_amp*y_base
        x_background += values[0]

        # 9. Calculate background function
        y_background = values[-1]*f(x_background, *values[0:-2])
        y_final = y_base + y_background
            
        # 10. Plots
        # base function plot
        if base_bool:
            p.line(x_base, y_base, line_width = 5, color = '#9D6C97', legend_label = 'base_function')
            
        # background function plot
        if background_bool:
            p.line(x_background, y_background, line_width = 5, color = '#9DD9C5', legend_label = 'background_function')
        
        # rough data plot
        if exp_bool:
            for k, col in enumerate(exp_data):
                p.line(x_rough, rough_df[col], legend_label = col, line_width = 5, color=color_palette[k+1])
                p.circle(x_rough, rough_df[col], legend_label = col, size = 7, color='#5F9545')

        # base + background
        if add_bool:
            indices = np.where(np.isin(x_base, x_rough+values[0]))[0]
            y_final_points = y_final[indices]
            p.line(x_base, y_final, line_width = 5, legend_label = 'Base + background functions', color = '#A6DDFF', alpha = 1.0)

        if downsampling_bool:    
            indices = np.where(np.isin(x_base, x_rough+values[0]))[0]
            y_final_points = y_final[indices]
            p.line(x_rough+values[0], y_final_points, line_width=5, legend_label = 'Downsampling', color = '#98473E',  alpha = 0.7)
            p.triangle(x_rough+values[0], y_final_points, size = 10, legend_label = 'Downsampling', color = '#DB8A74')

        p.x_range = Range1d(xlims[0], xlims[1])
        p.y_range = Range1d(ylims[0], ylims[1])
        p.xaxis.ticker.desired_num_ticks = 20
        p.yaxis.ticker.desired_num_ticks = 10
        vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
        p.add_layout(vline)
        p = plot_format(p, "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")
        with columns[1]:
            st.bokeh_chart(p)
        figures.append(p)

 


