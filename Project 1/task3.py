import numpy as np
import matplotlib.pyplot as plotter
from scipy.linalg import norm
import functions as f
import time

a, b, c, d = 3, 9, 15, 15

u0 = np.array([[1], [1]])  # different initial values between plots

t0 = 0
tf = 11  # or 11 * 100 for error plot

tol = 1e-10  # or 1e-6 for error plot


def lotka(t, u):
	x = a * u[0] - b * u[0] * u[1]
	y = c * u[0] * u[1] - d * u[1]
	return np.array([x, y])


def H(u):
	return c * u.item(0) + b * u.item(1) - d * np.log(u.item(0))\
		   - a * np.log(u.item(1))


H0 = H(u0)


def eq(h):
	return norm(h / H0 - 1)


hArr = np.array([eq(H0)])

ex = time.time()

t, u = f.adaptiveRK34(lotka, u0, t0, tf, tol)

for i in u[:, 1:].T:
	temp = H(i)
	temp = eq(temp)
	hArr = np.append(hArr, [temp])

ex = time.time() - ex
print(ex)

# I comment out different parts to produce the different plots

# plotter.plot(t, u[0], label='x')
# plotter.plot(t, u[1], label='y')
plotter.plot(u[0], u[1], label='phase portrait')
# plotter.semilogy(t, hArr, label='Error')
plotter.grid()
plotter.xlabel('x')  # or 'x' for phase portrait
plotter.ylabel('y') # for phase portrait
plotter.legend(loc='upper left')
plotter.show()
