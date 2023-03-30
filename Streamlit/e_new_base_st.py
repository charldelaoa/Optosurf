import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

# 1. Load data
# dfAnn = pd.read_csv("data/e/Ann_wafer10_calculated.csv")
raw_data = np.loadtxt("data/e/Ann_wafer7_off_diodes.dat", delimiter=',')
dfAnn = pd.read_csv("data/e/Ann_wafer7_calculate.csv")
raw_data_Ann_10 = np.loadtxt("data/e/Ann_wafer10_off_diodes.dat", delimiter=',')
na = 764
dfAnnbase = pd.read_csv("data/e/Ann_wafer10_calculated.csv")


# Get base function parameters
x_rough = np.arange(-15.5,16.5)
base_function = raw_data_Ann_10[na]
Aqbase = dfAnnbase.loc[na]['z']
weight_bool = True

# Def supergaussian
supergaussian = lambda x, x0, sigma, A1, n: A1 * np.exp(-abs(((x-x0)/sigma))**n)

# Def cost_function
def cost_function(params, y):
    x0, A0, sigma, A1, n, displacement = params
    # Get new x axis
    x_new = x_rough + x0
    # interpolate base function with respect to x_new (32 points)
    y_base_modified = A0*base_function
    # calculate background on original axis and with x0
    y_background = supergaussian(x_new, x0+displacement, sigma, A1, n)
    # calculate modified function
    y_modified = y_base_modified + y_background
    # Compare directly with 32 points experimental data
    if weight_bool:
        mse = np.mean(np.abs(x_rough)*((y - y_modified) ** 2))
        rmse = np.sqrt(mse)
    else:
        mse = np.mean((y - y_modified) ** 2)
        rmse = np.sqrt(mse)
    return rmse

# 2. Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# 3. Set up layout
app.layout = dbc.Container([
dbc.Row([
html.Div('Ann7 Plot', className="text-primary text-center fs-3")
]),
dbc.Row([
    dbc.Col([
        dcc.Graph(
            id='plot1',
            figure = go.Figure(data=go.Scatter(
                x=dfAnn['x'],
                y=dfAnn['y'],
                mode='markers',
                marker=dict(color=dfAnn['z'], colorscale='turbo'),
                customdata=dfAnn[['z', 'n']],
                hovertemplate='x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>z: %{customdata[0]:.2f}<br>n: %{customdata[1]}'
            )).update_layout(
                height=700,
                width=700,
                xaxis_title='x (mm)',
                yaxis_title='y (mm)',
                title='Ann7',
                coloraxis_colorbar=dict(title='z', len=0.75, y=0.5)
            )
        )
    ], width=12),
]),

dbc.Row([
    html.Div([
        dcc.Graph(id='subplots')
    ], className='col-12')
]),

dbc.Row([
    html.Div([
        html.H4('Selected Data'),
        html.Table(id='selected_data')
    ], className='col-12')
]),
], fluid=True)

# 4. Define callbacks
@app.callback(
    [Output('selected_data', 'children'), Output('subplots', 'figure')],
    Input('plot1', 'selectedData')
)


def display_selected_data(selectedData):
    if selectedData is not None:
        # Get selected data df
        indices = [point['pointIndex'] for point in selectedData['points']]
        selected_dfAnn = dfAnn.iloc[indices]
        selected_dfAnn = selected_dfAnn.sort_values(by=['z'])
        selected_data_table = [html.Tr([html.Th(col) for col in selected_dfAnn.columns])] + [html.Tr([html.Td(selected_dfAnn.iloc[i][col]) for col in selected_dfAnn.columns]) for i in range(len(selected_dfAnn))]
        
        # Get Ann10 data
        x_rough = np.arange(-15.5,16.5)
        base_function = raw_data_Ann_10[na]
        Aqbase = dfAnnbase.loc[na]['z']
        
        # Calculate the rows and cols of subplots
        n_unique = len(selected_dfAnn['n'].unique())
        cols = min(n_unique, 4)  # Maximum of 3 columns
        rows = (n_unique - 1) // cols + 1
        titles = list(selected_dfAnn['z'])
        fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05, subplot_titles=titles)
        print('New dataset')
        # Add scatter plots to each subplot
        for i, n in enumerate(selected_dfAnn['n']): 
            y_rough = raw_data[n]
            Aq = selected_dfAnn.loc[n]['z']

            # Minimization function
            guess = [0.3, 0.1, 1.0, 100.0, 2.0, 0.1]
            cost_fn = lambda p:cost_function(p, y_rough)
            result = minimize(cost_fn, guess)
            optimized_parameters = result.x
            x0_opt, A0_opt, sigma_opt, A1_opt, n_opt, displacement_opt = optimized_parameters
            print(f'Aq: {Aq:.4f}')
            print(f'x0_opt: {x0_opt}')
            print(f'A0_opt: {A0_opt}')
            print(f'sigma_opt: {sigma_opt}')
            print(f'A1_opt: {A1_opt}')
            print(f'n_opt: {n_opt}')
            print(f'displacement_opt: {displacement_opt}')

            # Calculate optimized function
            x_new_opt = x_rough + x0_opt
            y_base_opt = A0_opt*base_function
            y_background_opt = supergaussian(x_new_opt, x0_opt+displacement_opt, sigma_opt, A1_opt, n_opt)
            x_interp = np.arange(-15.5, 15.5001, 0.001).round(3)
            y_interp = supergaussian(x_interp, x0_opt+displacement_opt, sigma_opt, A1_opt, n_opt)
            area_back = np.trapz(y_interp, x=x_interp)
            print(f'area_background: {area_back}')
            y_optimized = y_base_opt + y_background_opt

            # Calculate error
            mse = np.mean(np.abs(x_rough)*((y_rough - y_optimized) ** 2))
            rmse = np.sqrt(mse)
            print(f'rmse: {rmse:.4f}')
            print('')

            # Plots
            # fig.add_trace(go.Scatter(x=x_rough, y=base_function, line = dict(width=4, color = '#7CB518'), name = f'original base; n: {n}',), 
            #               row=1+i//cols, col=1+i%cols)
            # fig.add_trace(go.Scatter(x=x_rough, y=base_function, mode='markers', name=f'original base points ({Aqbase:.3f})',marker=dict(size=14,symbol='triangle-up-dot', color = '#E08DAC')), 
            #               row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_base_opt, line = dict(width=4, color = '#DDB771'), name = f'base optimized; n: {n}',), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_base_opt, mode='markers', name=f'base optimized points ({Aqbase:.3f})',marker=dict(size=9, symbol='diamond', color = '#DDB771')), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_background_opt, line = dict(width=4, color = '#6BBF59'), name = f'background; n: {n}',), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_background_opt, mode='markers', name=f'background points ({Aqbase:.3f})',marker=dict(size=9, color = '#6BBF59')), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_rough, line = dict(width=4, color= '#3F7CAC', dash='dash'), name = f'Rough data Aq: {Aq:.2f}; n: {n}',), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_rough, mode='markers', name = f'Aq: {Aq:.2f}', marker = dict(size=11, color='#6A7FDB'),), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_optimized, line = dict(width=4, color= '#FB8F67', dash='dash'), name = f'Optimized',), 
                          row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x_rough, y=y_optimized, mode='markers', name = f'Optimized points', marker = dict(size=14, color='#FB8F67', symbol='triangle-up-dot'),), 
                          row=1+i//cols, col=1+i%cols)
            
    
            fig.update_yaxes(title_text=f"Intensity ({n})", row=1+i//cols, col=1+i%cols,)

        fig.update_layout(
        height=400*rows,
        width=550*cols,
        margin=dict(l=0, r=0, t=50, b=0),  # Adjust the margins to make room for the subplot titles
        template='plotly_white'
        )

        return selected_data_table, fig

    else:
        return '', {}

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)