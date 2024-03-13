import numpy as np
import matplotlib.pyplot as plotter
from scipy.integrate import BDF
import functions as f
import time

u = np.array([10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000,
			  10000, 15000])
steps1 = np.zeros([13])
steps2 = np.zeros(15)

y0 = np.array([[2], [0]])
yi = np.array([2, 0])
tol = 1e-6

ex = time.time()

for i in range(15):  # or range(13) when using the written solver
	def vdp(t, y):
		y1 = y[1]
		y2 = u[i] * (1 - y[0] ** 2) * y[1] - y[0]
		return np.array([y1, y2])


	# t1, y1 = f.adaptiveRK34(vdp, y0, 0, .07 * u[i], tol)
	bdf = BDF(vdp, 0, yi, .07 * u[i])  # built-in solver
	k = 0
	while bdf.t < .07 * u[i]:
		bdf.step()
		k += 1
	# steps1[i] = t1.size
	steps2[i] = k

ex = time.time() - ex
print(ex)

# Comment out different parts of the code for different plots,
# written solver vs. built-in solver

# plotter.loglog(u[:13], steps1)
plotter.loglog(u, steps2)
plotter.grid()
plotter.xlabel('\u03BC')
plotter.ylabel('Number of steps')
plotter.show()
