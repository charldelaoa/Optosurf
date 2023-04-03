import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

# 1. Load data
# dfAnn = pd.read_csv("data/e/Ann_wafer10_calculated.csv")
raw_data_Ann_7 = np.loadtxt("data/e/Ann_wafer7_off_diodes.dat", delimiter=',')
dfAnn_7 = pd.read_csv("data/e/Ann_wafer7_calculate.csv")
raw_data_Ann_10 = np.loadtxt("data/e/Ann_wafer10_off_diodes.dat", delimiter=',')
dfAnn_10 = pd.read_csv("data/e/Ann_wafer10_calculated.csv")
na = 8012

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
                x=dfAnn_7['x'],
                y=dfAnn_7['y'],
                mode='markers',
                marker=dict(color=dfAnn_7['z'], colorscale='turbo'),
                customdata=dfAnn_7[['z', 'n']],
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
        print(indices)
        selected_df = dfAnn_7.iloc[indices]
        selected_df = selected_df.sort_values(by=['z'])
        selected_data_table = [html.Tr([html.Th(col) for col in selected_df.columns])] + [html.Tr([html.Td(selected_df.iloc[i][col]) for col in selected_df.columns]) for i in range(len(selected_df))]
        
        # Get Ann10 data
        base_function = raw_data_Ann_10[na]
        Aqbase = dfAnn_10.loc[na]['z']

        # Calculate the rows and cols of subplots
        n_unique = len(selected_df['n'])
        cols = min(n_unique, 4)  # Maximum of 3 columns
        rows = (n_unique - 1) // cols + 1
        titles = list(selected_df['z'])
        fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05, subplot_titles=titles)

        # Add scatter plots to each subplot
        for i, n in enumerate(selected_df['n']):
            x = np.arange(-15.5,16.5)
            y = raw_data_Ann_7[n]
            Aq = selected_df.loc[n]['z']
            fig.add_trace(go.Scatter(x=x, y=base_function, line = dict(width=4, color = '#7CB518'), name = f'Aqbase {Aqbase:.4f}',), 
                          row=1+i//cols, col=1+i%cols)
            # fig.add_trace(go.Scatter(x=x, y=base_function, mode='markers', name=f'base points ({Aqbase:.3f})',marker=dict(size=14,symbol='triangle-up-dot', color = '#E08DAC')), 
            #               row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(x=x, y=y, line = dict(width=4, color= '#3F7CAC', dash='dash'), name = f'Aq: {Aq:.2f}; n: {n}',), 
                          row=1+i//cols, col=1+i%cols)
            # fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name = f'Aq: {Aq:.2f}', marker = dict(size=11, color='#6A7FDB'),), 
            #               row=1+i//cols, col=1+i%cols)
            
    
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