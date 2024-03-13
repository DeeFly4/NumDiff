import functions as f
import numpy as np
import matplotlib.pyplot as plotter

u = 100


def vdp(t, y):
	y1 = y[1]
	y2 = u * (1 - y[0] ** 2) * y[1] - y[0]
	return np.array([y1, y2])


y0 = np.array([[1], [2]])
t0 = 0
tf = 2 * u

tol = 1e-10

t, y = f.adaptiveRK34(vdp, y0, t0, tf, tol)

# Comment out different parts of the code for different plots

# plotter.plot(t, y[1, :], label=r'$y_2(t)$')
plotter.plot(y[0, :], y[1, :], label='phase portrait')
plotter.grid()
# plotter.xlabel('Time')
plotter.xlabel(r'$y_1$')
plotter.ylabel(r'$y_2$')
plotter.legend(loc='upper center')
plotter.title(r'$y(0) = (1, 2)^T$')
plotter.show()