import numpy as np
import functions as f
import time

a = -5
d = .01
mesh = 1
N = 500  # int(round(abs(1 / mesh * a / d)))
M = 500


def g(x):
	return np.exp(-100 * (x - .5) ** 2)


def twopulse(x):
	return np.exp(-100 * (x - .25) ** 2) + .7 * np.exp(-100 * (x - .75) ** 2)


x = np.linspace(0, 1, N + 1)

gvec = twopulse(x[:-1])

ex = time.time()

U = f.convdifsolver(gvec, a, d, N, M)

f.plotConvDif(U, N, M)

print(time.time() - ex)
