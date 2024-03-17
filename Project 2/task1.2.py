import numpy as np
import functions as f
import matplotlib.pyplot as plotter
import time

L = 10
E = 1.9e11
q = -50e3
N = 999

q = q * np.ones(N)  # q is constant along the beam
x = np.linspace(0, L, N + 2)


def ieq(x):  # calculates I(x)
	return 1e-3 * (3 - 2 * np.cos(np.pi * x / L) ** 12)

ex = time.time()

M = f.twopBVP(q, 0, 0, L, N)  # solving for M with 0 at boundaries

fvec = M[1:-1] / (E * ieq(x)[1:-1]) # creating fvec according to formula

u = f.twopBVP(fvec, 0, 0, L, N)  # solving for u with 0 at boundaries

ex = time.time() - ex # execution time
print(ex)
print(u[500].round(9))

plotter.plot(x, u)
plotter.grid()
plotter.show()
