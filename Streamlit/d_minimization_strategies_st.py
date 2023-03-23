import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Span, Range1d
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Set3
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
st.set_page_config(page_title="Minimization strategies", layout="wide")

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

# 1. Get base function points
new_colors = []
for i in range(42):
        new_colors.append('#9D6C97')
        new_colors.append('#9DC3E6')
        new_colors.append('#9DD9C5')

# a. Read the Excel file into a DataFrame
df = pd.read_excel('data/base_function.xlsx', sheet_name=['base', 'M'])

# b. Split the DataFrame into two separate DataFrames
base_df = df['base']
M_df = df['M'].sort_values(by='M')
sorted_df = pd.DataFrame(columns=['mu','xaxis', 'yaxis', 'colors'])

# c. Create x axis
xaxis = np.arange(-15.5, 16.5, 1)
plots = []

# d. Iterate M dataframe
for i, (index, row) in enumerate(M_df.iterrows()):
    # Create dataframe
    new_axis = xaxis - row.M
    sorted_df = sorted_df.append(pd.DataFrame({'mu':[row.M]*32,'xaxis':new_axis, 'yaxis':base_df[index], 'colors':new_colors[0:32]}), ignore_index=True)
    
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

# 1. Get base function points (330 points from -16.4 to 16.5)
x_base = smooth_df.xaxis.values
y_base = smooth_df.yaxis.values

# 2. get rough data 
rough_df = pd.read_excel('data/rough_samples.xlsx', sheet_name='Data')
source_rough = ColumnDataSource(rough_df)
x_rough = rough_df['xaxis'].values.round(3)
cols = rough_df.columns

# 3. Get initial guesses
calculate_bool = st.sidebar.checkbox("Calculate minimization", value=True)
weight_bool = st.sidebar.checkbox("Weighted minimization", value=True)
rough_data = st.sidebar.multiselect("Select rough data", list(cols[1:]), list(cols[1:2]))
limit = st.sidebar.number_input("Limit", value=30000, min_value=0, max_value=100000, step=1000)
methods = ['Powell', 'CG', 'L-BFGS-B', 'SLSQP', 'trust-constr']
methods_select = st.sidebar.multiselect("Select methods", methods, methods) 

st.sidebar.markdown(f"### Initial guesses for {rough_data[0]} (Can modify the values)")
guess_df_a = pd.read_excel('data/guesses.xlsx', sheet_name=rough_data[0])
guess_df_a = guess_df_a.set_index('Variables')
edited_guess = st.sidebar.experimental_data_editor(guess_df_a.loc[:][methods_select], num_rows="dynamic")

# 4. Create df that will save optmized parameters
# methods = ['Nelder-Mead', 'Powell', 'CG', 'L-BFGS-B', 'COBYLA', 'SLSQP', 'trust-constr']

# index = ['x0', 'Abase', 'sigma', 'Agaussian', 'n', 'displacement', 'error', 'area_background', 'area_modified']
index = ['x0', 'Abase', 'sigma', 'Agaussian', 'n', 'displacement', 'error', 'area_background']
optimized_df = pd.DataFrame(columns=methods_select, index=index)

# 5. Define gaussian function
supergaussian = lambda x, x0, sigma, A1, n: A1 * np.exp(-abs(((x-x0)/sigma))**n)

# 6. Define cost function
pchip = PchipInterpolator(x_base, y_base)
def cost_function(params, y):
    x0, A0, sigma, A1, n, displacement = params
    mask = y < limit
    y = y[mask]
    # Get new x axis
    x_new = x_rough[mask] + x0
    # interpolate base function with respect to x_new (32 points)
    y_base_modified = A0*pchip(x_new) 
    # calculate background on original axis and with x0
    y_background = supergaussian(x_new, x0+displacement, sigma, A1, n)
    # calculate modified function
    y_modified = y_base_modified + y_background
    # Compare directly with 32 points experimental data
    if weight_bool:
        mse = np.mean(np.abs(x_rough[mask])*((y - y_modified) ** 2))
        rmse = np.sqrt(mse)
    else:
        mse = np.mean((y - y_modified) ** 2)
        rmse = np.sqrt(mse)
    convergence.append(rmse)
    return rmse

# 7. Iterate over the experimental data
rough_plots = []
color_palette = Set3[10]
bounds = ((-0.3, 0.3), (-0.5, 1.2), (1, 4), (0, None), (0.1, 4), (-0.4, 0.4))
         #x0           #Abase        #sigma  #Agaussian  #n       #displacament
backgrounds = figure(title = f'Background functions all methods', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
differences = figure(title = f'Errors all methods', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
convergences = figure(title = f'Base function all methods', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
optimized = figure(title = f'Optimized functions all methods', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])

col = rough_data[0]
if calculate_bool:
    for j, method in enumerate(methods_select):
        convergence = []
        # 8. Get initial guesses
        rough_plot = figure(title = f"{col}: {method}", width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
        backgroundsbg = figure(title = f'Error - {method}', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
        differenceg = figure(title = f'Gaussian Experimental vs Optimized differences - {method}', width = 550, height = 450, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
        guess = []
        guess = [edited_guess.loc[var][method] for var in ['x0', 'Abase', 'sigma', 'Agaussian', 'n', 'displacement']]

        # 9. Call minimization function
        y_rough = rough_df[col].copy().values
        cost_fn = lambda p:cost_function(p, y_rough)
        result = minimize(cost_fn, guess, method=method, bounds=bounds)
        optimized_parameters = result.x
       
        result2 = minimize(cost_fn, optimized_parameters, method=method, bounds=bounds)
        optimized_parameters2 = result2.x
        x0_opt, A0_opt, sigma_opt, A1_opt, n_opt, displacement_opt = optimized_parameters2

        colu = col + '_opt'
        optimized_df.loc['x0'][method] = x0_opt
        optimized_df.loc['Abase'][method] = A0_opt
        optimized_df.loc['sigma'][method] = sigma_opt
        optimized_df.loc['Agaussian'][method] = A1_opt
        optimized_df.loc['n'][method] = n_opt
        optimized_df.loc['displacement'][method] = displacement_opt

        # 7. Calculate new optimized modified function
        mask = y_rough < limit
        y_rough = y_rough[mask]
        x_rough_n = x_rough[mask]
        # Get new x axis
        x_new_opt = x_rough[mask] + x0_opt
        # interpolate base function with respect to x_new (32 points)
        y_base_opt = A0_opt*pchip(x_new_opt) 
        # calculate background on original axis and with x0
        y_background_opt = supergaussian(x_new_opt, x0_opt+displacement_opt, sigma_opt, A1_opt, n_opt)
        optimized_df.loc['area_background'][method] = np.trapz(y_background_opt, x=x_new_opt)
        # calculate optmized function
        y_optimized = y_base_opt + y_background_opt
        # optimized_df.loc['area_modified'][method] = np.trapz(y_optimized, x=x_rough_n)

        # 8. Calculate error
        # if weight_bool:
        #     mse = np.mean(np.abs(x_rough[mask])*((y_rough - y_optimized) ** 2))
        #     rmse = np.sqrt(mse)
        # else:
        mse = np.mean((y_rough - y_optimized) ** 2)
        rmse = np.sqrt(mse)
        optimized_df.loc['error'][method] = rmse
        
        vline = Span(location=0.0, dimension = 'height', line_color='#FEEED9', line_width=1)
        rough_plot.add_layout(vline)

        # Plot optimize function lines
        rough_plot.line(x_rough_n, y_base_opt, legend_label = 'Base', line_width = 5, color='#F96F5D')
        rough_plot.line(x_rough_n, y_background_opt, legend_label = 'Bbackground', line_width = 5, color='#F9B5AC')
        rough_plot.line(x_rough_n, y_optimized, legend_label = 'Optimized function', line_width = 5, color='#987284')
        rough_plot.triangle(x_rough_n, y_optimized, legend_label = 'Optimized points', size = 8, color=color_palette[1])
        backgrounds.line(x_rough_n, y_background_opt, color = color_palette[j], line_width = 5 , legend_label = f"{method}")
        backgrounds.circle(x_rough_n, y_background_opt, fill_color = color_palette[j], size = 7 , legend_label = f"{method}")
        backgroundsbg.line(np.arange(0, len(convergence)), convergence, color = color_palette[j], line_width = 5 , legend_label = f"Background {col}")
        # downsamplesg.line(x_rough_n, y_optimized, line_width=4, legend_label = f'Downsampling {col}', color = color_palette[i+1],  alpha = 0.9, line_dash='dashed')
        # downsamplesg.triangle(x_rough, y_optimized, size = 13, legend_label = f'Downsampling {col}', color = color_palette[i+1])

        # Plot rough experimental data
        rough_plot.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4, line_dash = 'dashed')
        rough_plot.circle('xaxis', col, source=source_rough, fill_color= color_palette[j], size=7, legend_label = f"{col} points")
        
        # Error convergence plot
        convergences.line(x_rough_n, y_base_opt, legend_label = method, color=color_palette[j], line_width=5)

        optimized.line(x_rough_n, y_optimized, legend_label = method, color=color_palette[j], line_width=5)
        optimized.triangle(x_rough_n, y_optimized, legend_label = method, size = 8, color=color_palette[1])

        # Plot format
        rough_plot.y_range = Range1d(-5000, 50000)
        rough_plot.xaxis.ticker.desired_num_ticks = 10
        rough_plot.yaxis.ticker.desired_num_ticks = 10
        rough_plot = plot_format(rough_plot, "Degrees", "Intensity", "top_left", "10pt", "8pt", "9pt")
        # rough_plots.append(rough_plot)

        # Difference plot
        diff = y_rough - y_optimized
        differenceg.line(x=x_rough_n, y=diff, legend_label = col, color = color_palette[j], line_width=4)
        differenceg.circle(x=x_rough_n, y=diff, legend_label = col, fill_color= color_palette[j], size=7)
        differences.line(x=x_rough_n, y=diff, legend_label = method, color = color_palette[j], line_width=4)
        differences.circle(x=x_rough_n, y=diff, legend_label = method, fill_color = color_palette[j], size=6)
        
        plots = [differenceg, backgroundsbg]
        for plot in plots:
            plot = plot_format(plot, "Degrees", "Intensity", "top_left", "10pt", "10pt", "9pt")
            # rough_plots.append(plot)
            plot.xaxis.ticker.desired_num_ticks = 10
            plot.yaxis.ticker.desired_num_ticks = 10
        differenceg.y_range = Range1d(-300, 300)
        # backgrounds.y_range = Range1d(-2000, 10000)
        backgrounds.add_layout(vline)
        backgroundsbg.add_layout(vline)

    optimized.line('xaxis', col, source=source_rough, color = '#9DC3E6', legend_label = str(col), line_width=4, line_dash = 'dashed')
    optimized.circle('xaxis', col, source=source_rough, fill_color= color_palette[j], size=7, legend_label = f"{col} points")

    st.sidebar.write(optimized_df)
    backgrounds = plot_format(backgrounds, "Degrees", "Intensity", "top_left", "9pt", "9pt", "9pt")
    differences = plot_format(differences, "Degrees", "Intensity", "top_left", "9pt", "9pt", "9pt")
    convergences = plot_format(convergences, "Degrees", "Intensity", "top_left", "9pt", "9pt", "9pt")
    optimized = plot_format(optimized, "Degrees", "Intensity", "top_left", "9pt", "9pt", "9pt")

    # convergences.y_range = Range1d(-2000, 50000)
    differences.y_range = Range1d(-2000, 2000)
    rough_plots.insert(0, backgrounds)
    rough_plots.insert(1, differences)
    rough_plots.insert(2, convergences)
    rough_plots.insert(3, optimized)

    grid_rough = gridplot(children = rough_plots, ncols = 2, merge_tools=False, width = 500, height = 340)
    st.bokeh_chart(grid_rough)