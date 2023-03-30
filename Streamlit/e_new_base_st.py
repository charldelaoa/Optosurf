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
raw_data = np.loadtxt("data/e/Ann_wafer10_off_diodes.dat", delimiter=',')
dfAnn = pd.read_csv("data/e/Ann_wafer10_calculated.csv")

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# Set up layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('Ann10 Plot', className="text-primary text-center fs-3")
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='plot1',
                figure=go.Figure(data=go.Scatter(
                    x=dfAnn['x'],
                    y=dfAnn['y'],
                    mode='markers',
                    marker=dict(color=dfAnn['z'], colorscale='turbo'),
                    customdata=dfAnn[['z', 'n']],
                    hovertemplate='x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>z: %{customdata[0]:.2f}<br>n: %{customdata[1]}'
                )).update_layout(
                    height=600,
                    width=600,
                    xaxis_title='x (mm)',
                    yaxis_title='y (mm)',
                    title = 'Ann10',
                    coloraxis_colorbar=dict(title='z', len=0.75, y=0.5)
                )
            )
        ], width=4),

        dbc.Col([
            dcc.Graph(id='subplots')
        ], width=6),
    ]),

    html.Div([
        html.H4('Selected Data'),
        html.Table(id='selected_data')
    ], className='col-12'),
], fluid=True)

# Define callbacks
@app.callback(
    [Output('selected_data', 'children'), Output('subplots', 'figure')],
    Input('plot1', 'selectedData')
)
def display_selected_data(selectedData):
    if selectedData is not None:
        indices = [point['pointIndex'] for point in selectedData['points']]
        selected_dfAnn = dfAnn.iloc[indices]
        selected_data_table = [html.Tr([html.Th(col) for col in selected_dfAnn.columns])] + [html.Tr([html.Td(selected_dfAnn.iloc[i][col]) for col in selected_dfAnn.columns]) for i in range(len(selected_dfAnn))]
        n_unique = len(selected_dfAnn['n'].unique())
        cols = min(n_unique, 3)  # Maximum of 3 columns
        rows = (n_unique - 1) // cols + 1

        fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05)

        # Add scatter plots to each subplot
        for i, n in enumerate(selected_dfAnn['n'].unique()):
            n_data = raw_data[n]
            x = np.arange(-15.5,16.5)
            y = n_data
            Aq = selected_dfAnn.loc[n]['z']
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name = f'Aq: {Aq:.2f}'
                ), row=1+i//cols, col=1+i%cols)
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                ), row=1+i//cols, col=1+i%cols)
            fig.update_xaxes(
            title_text=f"Aq: {Aq:.4f}",
            row=1+i//cols,
            col=1+i%cols,
            )

            
            # fig.add_trace(go.Scatter(
            # x=selected_dfAnn[selected_dfAnn['n']==n]['x'],
            # y=selected_dfAnn[selected_dfAnn['n']==n]['y'],
            # mode='markers',
            # marker=dict(size=2, color='black'),
            # showlegend=False,
            # ), row=1+i//5, col=1+i%5)   

        fig.update_layout(
        height=400*rows,
        width=550*cols,
        margin=dict(l=0, r=0, t=50, b=0)  # Adjust the margins to make room for the subplot titles
        )

        return selected_data_table, fig

    else:
        return '', {}

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)