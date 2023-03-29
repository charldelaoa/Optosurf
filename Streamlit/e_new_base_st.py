
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd

# Load mtcars dataset
mtcars = pd.read_csv('https://plotly.github.io/datasets/mtcars.csv')

# Define app
app = dash.Dash(__name__)

# Set up layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='plot1',
            figure=px.scatter(mtcars, x='wt', y='mpg')
        )
    ], className='col-4'),
    html.Div([
        dcc.Graph(id='plot2')
    ], className='col-12'),
    html.Div([
        html.H4('Selected Data'),
        html.Table(id='selected_data')
    ], className='col-12')
], className='row')

# Define callbacks
@app.callback(
    Output('plot2', 'figure'),
    Input('plot1', 'selectedData')
)
def display_brushed_plot(selectedData):
    if selectedData is not None:
        indices = [point['pointIndex'] for point in selectedData['points']]
        selected_mtcars = mtcars.iloc[indices]
        try:
            fig = px.scatter(selected_mtcars, x='disp', y='hp')
        except Exception as e:
            print(str(e))
            fig = None
        return fig
    else:
        return {}

@app.callback(
    Output('selected_data', 'children'),
    Input('plot1', 'selectedData')
)
def display_selected_data(selectedData):
    if selectedData is not None:
        indices = [point['pointIndex'] for point in selectedData['points']]
        selected_mtcars = mtcars.iloc[indices]
        return [html.Tr([html.Th(col) for col in selected_mtcars.columns])] + [html.Tr([html.Td(selected_mtcars.iloc[i][col]) for col in selected_mtcars.columns]) for i in range(len(selected_mtcars))]
    else:
        return []

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)