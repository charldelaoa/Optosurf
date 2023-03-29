import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# 1. Load data
raw_data = np.loadtxt("data/e/Ann_wafer10_off_diodes.dat", delimiter=',')
dfAnn = pd.read_csv("data/e/Ann_wafer10_calculated.csv")

# Define app
app = dash.Dash(__name__)

# Set up layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='plot1',
            figure=go.Figure(data=go.Scatter(
                x=dfAnn['x'],
                y=dfAnn['y'],
                mode='markers',
                marker=dict(color=dfAnn['z'], colorscale='turbo'),
                customdata=dfAnn[['z', 'radii', 'angle', 'n']],
                hovertemplate='x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>z: %{customdata[0]:.2f}<br>radii: %{customdata[1]:.2f}<br>angle: %{customdata[2]:.2f}<br>n: %{customdata[3]}'
            )).update_layout(
                height=600,
                width=600,
                xaxis_title='x (mm)',
                yaxis_title='y (mm)',
                title = 'Ann10',
                coloraxis_colorbar=dict(title='z', len=0.75, y=0.5)
            )
        )
    ], className='col-12'),
    html.Div([
        html.H4('Selected Data'),
        html.Table(id='selected_data')
    ], className='col-12'),
    dcc.Graph(id='subplots')
], className='row')

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

        # Create a grid of subplots for each index n in the selected data
        fig = make_subplots(rows=4, cols=8, subplot_titles=[f"Index n={n}" for n in selected_dfAnn['n'].unique()])

        # Add scatter plots to each subplot
        for i, n in enumerate(selected_dfAnn['n'].unique()):
            n_data = raw_data[n]
            x = np.arange(1,31)
            y = n_data
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                ), row=1+i//8, col=1+i%8)

        # Update the layout of the subplots
        fig.update_layout(
            height=800,
            width=1200,
            title=f"Index n={n}",
        )

        return selected_data_table, fig

    else:
        return '', {}

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)