from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import random
import pandas as pd
import time
from tkinter import *
import tkinter as tk

# Create initial data
x = []
y = []

# Create a ColumnDataSource object to hold the data
source = ColumnDataSource(data=dict(x=x, y=y))

# Create the plot and add a line renderer
plot = figure(title='Real-time data', plot_height=400, plot_width=800, tooltips = [("index", "$index"),("(x,y)", "($x, $y)")])
plot.line(x='x', y='y', source=source)
plot.circle(x='x', y='y', source=source)
df = pd.DataFrame(columns=['x', 'y'])
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
    df.to_csv(file, index=False)
# Add the plot to the document and define the update interval
counter = 0

#SETTING UP FILE NAME WINDOW
def closewindow():
    root.destroy()
 
def SaveFileName():
    FileName = str(FileNameEntry.get())
    FileNameLabel.destroy()
    FileNameEntry.destroy()
    FileNameButton.destroy()
 
root = Tk() #opening main window
 
root.wm_attributes("-topmost", 1)
root.title("1500-1600 Sweep") # naming window
root.geometry("700x250") # window size
 
#Setting label,entry path and save button for file name
FileName = tk.StringVar()
FileNameEntry = Entry(root, width = 50, border = 5 , textvariable = FileName)
FileNameLabel = Label(root, text = "File Name: ")
FileNameButton = Button(root, text = "Save" , command = SaveFileName)
CloseButton = Button(root, text = "Close" , fg = "red" , command = closewindow)
 
 
 
FileNameLabel.grid(row=0,column=0)
FileNameEntry.grid(row=0,column=1)
FileNameButton.grid(row=0,column=2)
CloseButton.grid(row=5,column=0)
 
root.mainloop()

file = FileName.get()

curdoc().add_root(plot)
curdoc().add_periodic_callback(update_data, 100)