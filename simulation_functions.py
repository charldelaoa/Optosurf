
def plot_format(plot, xlabel, ylabel, location, size, titlesize, labelsize):
    # x axis format
    plot.xaxis.axis_label = xlabel
    plot.xaxis.axis_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_style = "bold"
    plot.xaxis.axis_label_text_font_size = size
    plot.xaxis.major_label_text_font_size = size

    # y axis format
    plot.yaxis.axis_label = ylabel
    plot.yaxis.axis_label_text_font_style = 'bold'
    plot.yaxis.major_label_text_font_style = "bold"
    plot.yaxis.axis_label_text_font_size = size
    plot.yaxis.major_label_text_font_size = size

    # Legend format
    plot.legend.location = location
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = labelsize
    plot.legend.label_text_font_style = 'bold'
    plot.legend.border_line_width = 3
    plot.legend.background_fill_alpha = 0.0
    plot.legend.label_text_color = "#E3F4FF"
    # plot.legend.border_line_color = "navy"
    plot.legend.border_line_alpha = 0.5

    # Title format
    plot.title.text_font_size = titlesize

    plot.background_fill_color = "#0E1117"
    plot.border_fill_color = "#0E1117"

    plot.xgrid.grid_line_color = '#2D3135'
    plot.ygrid.grid_line_color = '#2D3135'
    
    plot.yaxis.major_label_text_color = "#E3F4FF"
    plot.xaxis.major_label_text_color = "#E3F4FF"
    plot.yaxis.axis_label_text_color = "#E3F4FF"
    plot.xaxis.axis_label_text_color = "#E3F4FF"
    plot.title.text_color = "#A6DDFF"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "15pt"

    plot.legend.click_policy="hide"
    return plot

def plot_equation(mu, sigma, n, number_points, degrees, plot, title="Super-Gaussian", width = 700, height = 550):
    """
    Plot the optical field equation using Bokeh

    Parameters
    ----------
    mu (float): Mean value for the equation
    sigma (float): Standard deviation value for the equation
    n (float): Order value for the equation
    number_points (int): number of points to calculate the function
    
    Returns
    -------
    plots (bokeh plot): Plot of the Super-Gaussian equation
    x(np): linspace for the gaussian plot
    y(np): gaussian values
    """
    # 1. Define linear degrees vector and calculate Super-Gaussian
    ticker = SingleIntervalTicker(interval=2.5, num_minor_ticks=10)
    xaxis = LinearAxis(ticker = ticker)
    x = np.linspace(degrees[0], degrees[1], number_points)
    y = np.exp(-abs(((x-mu)/sigma))**n)
    
    # 2. Plot 
    if plot:
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label='x', y_axis_label='y', tooltips = TOOLTIPS,
            width = width, height = height)
        p.line(x, y, line_width=4, alpha = 0.5, line_color = "#C5E064")
        p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        p = plot_format(p, "Degrees", "Intensity", "bottom_left", "10pt", "10pt", "10pt")
        return p, x, y
    else:
        return x, y


def window_integration(number_windows, window_size, gap, x, y, p=None):
    """
    Performs a window integration

    Parameters
    ----------
    number_windows (int): Number of integration windows
    window_size (int): Number of data points in the window
    x(np): linspace for the gaussian plot
    y(np): gaussian values
    Returns
    -------
    p (bokeh plot): Plot of the integration
    integration_axis (np): window integration axis
    integration_points (np): Integrated points
    """
    integration_points = []
    integration_axis = []
    color_multiplier = len(bokeh.palettes.Viridis256)//number_windows
    count = 0
    
    for i in range(number_windows):
    # 1. Get data in every window and integrate
        a = i*window_size
        b = i*window_size + window_size
        
        x_temp = x[a:b-gap:1]
        y_temp = y[a:b-gap:1]
        integration = np.trapz(y_temp, x_temp, dx = x[1] - x[0])
        integration_points.append(integration)

        axis = x_temp[len(x_temp)//2]
        integration_axis.append(axis)

        # 2. Draw a rectangle of the integration window
        if p is not None:
            left_edge = x_temp[0]
            right_edge = x_temp[-1]
            p.rect(x=(left_edge + right_edge)/2, y=0.18, width=right_edge-left_edge, height=0.3, fill_alpha=0.001, fill_color='#C5E0B4', color='#C5E0B4')
            p.rect(x=(right_edge + x[b-1])/2, y=0.18, width=x[b-1]-right_edge, height=0.3, fill_alpha=0.005, fill_color='#F16C08', color = '#F16C08')
            p.circle(x_temp[::15], y_temp[::15], size = 4, alpha = 1)
            count += 1
    if p is not None:
        p.circle(integration_axis, integration_points, size = 7, color = '#FAA0A0')
        p.line(integration_axis, integration_points, line_width = 4, color = '#FAA0A0', alpha = 0.8)
    return p, integration_axis, integration_points


def histogram_reconstruction(int_points, hist_bool, title="Super-Gaussian", width = 700, height = 550):
    """
    Constructs a histrogram

    Parameters
    ----------
    int_points(np): Points calculated from the window integration
    Returns
    -------
    hist_plot (bokeh plot): Plot of the histogram
    std_dev(float): Histogram's standard deviation
    """
    
    # a. Histogram reconstruction
    normalized_y = np.multiply(int_points, 10000)
    hist_2d = np.array([])
    for i, int_point in enumerate(normalized_y):
        round_int_point = round(float(int_point))
        hist_2d = np.concatenate((hist_2d, np.array([float(i)]*round_int_point)))
    
    # b. Calculate standard deviation
    stddev = np.std(hist_2d)
    
    # c. Plot histogram
    if hist_bool:
        title = f"Std_dev: {stddev:.5f}; " + title
        hist_plot = figure(title = title, plot_width=width, plot_height=height, x_axis_label='x', y_axis_label='count()')
        hist, edges = np.histogram(hist_2d, bins=20)
        hist_plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='#2D908C', line_color='black', alpha=0.7)
        hist_plot = plot_format(hist_plot, "x", "Count", "bottom_left", "10pt", "10pt", "10pt")
        return hist_plot, np.std(stddev)

    else:
        return stddev