import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np
import pandas as pd
# Create a Streamlit app
st.set_page_config(page_title="Super-Gaussian Equation Plotter", layout="wide")


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
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"
    # plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.0

    # Title format
    plot.title.text_font_size = titlesize

    plot.background_fill_color = "#0E1117"
    plot.border_fill_color = "#0E1117"

    plot.xgrid.grid_line_color = '#2D3135'
    plot.ygrid.grid_line_color = '#2D3135'
    
    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "15pt"

    plot.legend.click_policy="hide"
    return plot



# 1. Define the functions to be plotted
functions = [
    ("Gaussian", lambda x, x0, sigma: np.exp(-((x-x0)/sigma)**2/2), (0.0, 1.0), ('x0g', 'sigmag')),
    ("Lorentzian", lambda x, x0, sigma: np.exp(-((x-x0)/sigma)**2/2), (0.0, 1.0), ('x0l', 'sigmal')),
    ("Pseudo-Voigt", lambda x, x0, sigma, alpha: (1 - alpha) * np.exp(-((x-x0)/sigma)**2/2) + alpha/(1 + ((x-x0)/sigma)**2), (0.0, 0.5, 1.0), ('x0p', 'sigmap', 'alphap')),
    ("Squared cosine", lambda x, x0, c: np.where(np.abs(x-x0) <= c, 0.5*(1 + np.cos(np.pi*(x-x0)/c)), 0), (0.0, 2.0), ('x0cos', 'c'))
]
#  ("Squared cosine", lambda x, x0, c: np.where(np.abs(x-x0) <= c, 0.5*(1 + np.cos(np.pi*(x-x0)/c)), 0), (0.0, 2.0), ('x0cos', 'c'))
equations = [
    r"$\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$",
    r"$\frac{1}{1+\left(\frac{x-x_0}{\gamma}\right)^2}$",
    r"$(1-\gamma)\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right) + \frac{\gamma}{1+\left(\frac{x-x_0}{\sigma}\right)^2}$",
    r"$0.5*(1 + \cos(\pi\frac{(x-x_0)}{c}))$"
]

figures = [figure(title=function[0], width = 650, height = 400) for function in functions]
x = np.arange(-15.5, 15.5001, 0.001)

# 2. Get the base function
base_function = pd.read_csv('data/base_funtion_interpolated.csv')
rough_df = pd.read_excel('data/rough_samples.xlsx')
columns = rough_df.columns

# 2. Iterate over the lambda functions to create a slider per parameter
st.sidebar.title("Function parameters")
Amp = st.sidebar.slider("Base function amplitude", 0.0, 1.0, 1.0, 0.1)
x_base = base_function['x_base']
y_base = Amp*base_function['y_base']
A = st.sidebar.slider("Amplitude", 0, 50000, 35000, 1000)
for j, (name, f, params_nums, params_names) in enumerate(functions):
    # a. Add eq. in latex format and boolean to plot
    st.sidebar.title(name)
    st.sidebar.markdown(equations[j])
    plot_bool = st.sidebar.checkbox(f'Plot {name}', True)
    
    # b. Iterate over each parameter to create a slider
    values = []
    for i, param in enumerate(params_names):
        value = st.sidebar.slider(param, 0.0, 10.0, params_nums[i], 0.5)
        values.append(value)

    # c. Evaluate function
    y = A*f(x, *values)
    y_sub = y_base + y
    # d. Add plot
    figures[j].line(x_base, y_base, line_width = 5, color = '#9D6C97', legend_label = 'base_function', line_dash = 'dashed')
    figures[j].line(x, y, line_width = 5, color = '#9DD9C5', legend_label = name, line_dash = 'dashdot')
    figures[j].line(x, y_sub, line_width = 5, legend_label = 'background', line_dash = 'dashdot')
    figures[j].line(rough_df['xaxis'], rough_df['pt2d'], legend_label = 'pt2d', line_width = 5, color='#9DC3E6')
    figures[j].circle(rough_df['xaxis'], rough_df['pt2d'], legend_label = 'pt2d', size = 7)
    figures[j] = plot_format(figures[j], "Degrees", "Intensity", "top_left", "10pt", "10pt", "10pt")

grid = gridplot(children=figures, ncols = 2, merge_tools=False)
st.bokeh_chart(grid)


