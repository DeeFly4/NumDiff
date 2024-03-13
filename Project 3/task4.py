import functions as f
import numpy as np
import time

tf = 1
d = .005
N = 300
M = tf * 2000


def g(x):
	return np.exp(-100 * (x - .5) ** 2)


def twopulse(x):
	return np.exp(-200 * (x - .25) ** 2) + .7 * np.exp(-200 * (x - .75) ** 2)


x = np.linspace(0, 1, N + 1)

gvec = 5 * twopulse(x[:-1])

ex = time.time()

U = f.burgerssolver(gvec, d, N, M, tf)

f.plotConvDif(U, N, M, tf)

print(time.time() - ex)
