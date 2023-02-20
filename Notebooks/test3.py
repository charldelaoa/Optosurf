from bokeh.io import curdoc, show
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
plot = figure(title='Real-time data', plot_height=400, plot_width=800, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
plot.line(x='x', y='y', source=source)
plot.circle(x='x', y='y', source=source)

# Define the callback function to update the plot with new data
def update_data():
    global counter, df
    if counter >= 10:
        show(plot)
        curdoc().clear()
        return
    
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

# Add the plot to the document and define the update interval
counter = 0
curdoc().add_root(plot)
curdoc().add_periodic_callback(update_data, 100)

server = show(plot, notebook_url='localhost:8888')
server.stop()  # stop the server programatically