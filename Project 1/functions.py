import numpy as np
from scipy.linalg import expm, norm, solve
import matplotlib.pyplot as plotter


def eulerStep(A, uold, h):
	return uold + h * A@uold


def iEulerStep(A, uold, h):
	temp = np.eye(np.size(uold)) - h * A
	return solve(temp, uold)


def RK4Step(f, uold, told, h):
	y1 = f(told, uold)
	y2 = f(told + h / 2, uold + h / 2 * y1)
	y3 = f(told + h / 2, uold + h / 2 * y2)
	y4 = f(told + h, uold + h * y3)

	return uold + h / 6 * (y1 + 2 * y2 + 2 * y3 + y4)


def eulerInt(A, y0, t0, tf, N):
	h = (tf - t0) / N
	uold = y0
	approx = np.zeros((np.size(y0), N + 1))
	err = np.zeros(N + 1)
	approx[:, 0] = y0[:, 0]
	err[0] = 0

	for i in range(1, N + 1):
		unew = eulerStep(A, uold, h)
		approx[:, i] = unew[:, 0]
		exact = expm((t0 + h * i) * A)@y0
		err[i] = norm(exact[:, 0] - unew[:, 0])
		uold = unew

	return approx, err


def iEulerInt(A, y0, t0, tf, N):
	h = (tf - t0) / N
	uold = y0
	approx = np.zeros((np.size(y0), N + 1))
	err = np.zeros(N + 1)
	approx[:, 0] = y0[:, 0]
	err[0] = 0

	for i in range(1, N + 1):
		unew = iEulerStep(A, uold, h)
		approx[:, i] = unew[:, 0]
		exact = expm((t0 + h * i) * A)@y0
		err[i] = norm(exact[:, 0] - unew[:, 0])
		uold = unew

	return approx, err


def errVSh(A, y0, t0, tf, k=10):
	N = np.arange(k)
	N = np.power(2, N)

	h = (tf - t0) / N
	errors = np.zeros(k)

	for i in range(k):
		approx, err = eulerInt(A, y0, t0, tf, N[i])
		errors[i] = err[-1]

	plotter.loglog(h, errors)
	plotter.show()


def iErrVSh(A, y0, t0, tf, k=10):
	N = np.arange(k)
	N = np.power(2, N)

	h = (tf - t0) / N
	errors = np.zeros(k)

	for i in range(k):
		approx, err = iEulerInt(A, y0, t0, tf, N[i])
		errors[i] = err[-1]

	plotter.loglog(h, errors)
	plotter.show()


def RK34Step(f, uOld, tOld, h):
	y1 = f(tOld, uOld)
	y2 = f(tOld + h / 2, uOld + h / 2 * y1)
	y3 = f(tOld + h / 2, uOld + h / 2 * y2)
	y4 = f(tOld + h, uOld + h * y3)

	z3 = f(tOld + h, uOld - h * y1 + 2 * h * y2)

	return uOld + h / 6 * (y1 + 2 * y2 + 2 * y3 + y4), \
		   h / 6 * (2 * y2 + z3 - 2 * y3 - y4)


def newStep(tol, err, errOld, hOld, k):
	r1 = norm(err)
	r0 = norm(errOld)

	return (tol / r1) ** (2 / (3 * k)) * (tol / r0) ** (-1 / (3 * k)) * hOld


def adaptiveRK34(f, y0, t0, tf, tol):
	t = np.array([t0])  # array of time points, times will be appended in loop
	y = y0  # array of values corresponding to time points in array above, values
	# will be appended in loop

	if np.size(y0) == 1:
		axis = None
	else:
		axis = 1

	tOld = t0  # initial time, updates every loop
	yOld = y0  # initial value, updates every loop
	errOld = tol  # initial error set to tol, updates every loop

	hOld = ((tf - t0) * tol ** .25) / (100 * (1 + norm(f(t0, y0))))
	# initial step size, updates every loop

	while (tOld + hOld) < tf:  # if the desired step size stays within
		# time boundary tf
		t = np.append(t, [tOld + hOld])
		# appends the current time after the step to time array
		yOld, err = RK34Step(f, yOld, tOld, hOld)
		# calculates value and error after step and updates
		y = np.append(y, yOld, axis)  # appends the new value to value array

		tOld += hOld  # updates the current time to time after the step
		hOld = newStep(tol, err, errOld, hOld, 4)
		# calculates and updates step size for next step
		errOld = err  # updates the error at current time for next loop

	hLast = tf - tOld  # length of last step is simply the remaining distance
	t = np.append(t, tf)  # we know the last time value is tf
	yOld, err = RK34Step(f, yOld, tOld, hLast)  # calculates the value at time tf
	y = np.append(y, yOld, axis)  # appends the last value

	return t, y  # returns time and value array
