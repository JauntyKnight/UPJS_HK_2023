import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# read the data
ys = []
with open('./outputs/anomalies.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        ys.append(float(row[0]))


# plot the data
Plot, Axis = plt.subplots()

xs = list(range(len(ys)))

l = plt.plot(xs, ys, 'b-')

slider_color = 'White'

# Set the axis and slider position in the plot
axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                         facecolor = slider_color)
slider_position = Slider(axis_position,
                         'Pos', 0.1, 90.0)

# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    Axis.axis([pos, pos+10, -1, 1])
    Plot.canvas.draw_idle()
 
# update function called using on_changed() function
slider_position.on_changed(update)
 
# Display the plot
plt.show()