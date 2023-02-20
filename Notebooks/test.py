from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import random

# Create initial data
x = []
y = []
for i in range(50):
    x.append(i)
    y.append(random.random())

# Create a ColumnDataSource object to hold the data
source = ColumnDataSource(data=dict(x=x, y=y))

# Create the plot and add a line renderer
plot = figure(title='Real-time data', plot_height=400, plot_width=800)
plot.line(x='x', y='y', source=source)

# Define the callback function to update the plot with new data
def update_data():
    new_data = dict(x=[x[-1]+1], y=[random.random()])
    source.stream(new_data, rollover=50)

# Add the plot to the document and define the update interval
curdoc().add_root(plot)
curdoc().add_periodic_callback(update_data, 100)