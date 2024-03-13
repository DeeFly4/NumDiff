import numpy as np
import matplotlib.pyplot as plotter
import functions as f
from scipy.linalg import expm, norm

A = np.array([[-1, 10], [0, - 3]])


def eq(t, y):
	return np.dot(A, y)


y0 = np.array([[1], [1]])
t0 = 0
tf = 10
tol = 1e-8

t, y = f.adaptiveRK34(eq, y0, t0, tf, tol)

exact = y0
for i in t[1:]:
	exact = np.append(exact, np.dot(expm(A * i), y0), axis=1)

# Comment out different parts of the code for different plots

plotter.plot(t, y[0], label=r'$y_1$')
plotter.plot(t, y[1], label=r'$y_2$')
# plotter.plot(t, exact[0], label=r'$y_1$')
# plotter.plot(t, exact[1], label=r'$y_2$')
plotter.xlabel('Time')
plotter.title('Numerical solution')  # or 'Exact solution'
plotter.legend()
plotter.grid()
plotter.show()
