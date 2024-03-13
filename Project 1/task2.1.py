import numpy as np
import functions as func
from scipy.linalg import norm
import matplotlib.pyplot as plotter

a = 1 # or a = -1, ran it with different a to produce two plots


def eq(t, y):
	return a * y


y0 = 1
t0 = 0
tf = 1


def RK4Int(f, y0, t0, tf, N):
	h = (tf - t0) / N
	uold = y0
	told = t0
	approx = np.array([y0])
	err = np.array([0])

	for i in range(1, N + 1):
		unew = func.RK4Step(f, uold, told, h)
		approx = np.append(approx, [unew])
		exact = y0 * np.exp(a * h * i)
		err = np.append(err, norm(exact - unew))
		uold = unew
		told += h

	return approx, err


def errRK4(f, y0, t0, tf, k=10):
	N = np.arange(k)
	N = np.power(2, N)

	h = (tf - t0) / N

	errors = np.zeros(k)

	for i in range(k):
		approx, err = RK4Int(f, y0, t0, tf, N[i])
		errors[i] = err[-1]

	plotter.loglog(h, errors)
	plotter.grid()


errRK4(eq, y0, t0, tf)
plotter.xlabel('Step size')
plotter.ylabel('End point error')
plotter.title('a = 1')
plotter.savefig('a = 1')
plotter.show()
