from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import random
import pandas as pd
import time

# Define the callback function to update the plot with new data
def update_data():
    global counter, df
    if counter >= 10:
        curdoc().stop()

    counter += 1
    x.append(counter)
    y.append(counter)    
    new_data = dict(x=[x[-1]], y=[y[-1]])
    # time.sleep(0.5)
    print('counter: ', counter)
    print('new_data:', new_data)
    source.stream(new_data, rollover=50)
    df.loc[counter-1] = [x[-1], y[-1]]
    df.to_csv('data.csv', index=False)

# 1. Create initial data
x = []
y = []

# 2. Create a ColumnDataSource object to hold the data
source = ColumnDataSource(data=dict(x=x, y=y))

# 3. Create the plot and add a line renderer
plot = figure(title='Real-time data', plot_height=400, plot_width=800, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
plot.line(x='x', y='y', source=source)
plot.circle(x='x', y='y', source=source)
df = pd.DataFrame(columns=['x', 'y'])
counter = 0

# 4. Add the plot to the document and define the update interval
curdoc().add_root(plot)
curdoc().add_periodic_callback(update_data, 100)