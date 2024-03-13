import numpy as np
import functions as f
from scipy.linalg import norm
import matplotlib.pyplot as plotter
import time



def eq(x):
	return 6 * x


L = 10  # boundary point
N = 10000  # number of interior points
x = np.linspace(0, L, N + 2)

a = -2  # boundary value at 0
b = L ** 3 + L - 2  # boundary value at L

h = L / (N + 1)  # gap between interior points

v = eq(h * np.arange(1, N + 1))

ex = time.time()

y = f.twopBVP(v, a, b, L, N)

exact = x ** 3 + x - 2

err = np.zeros(N + 2)

for i in range(N + 1):
	err[i] = norm(y[i] - exact[i])

ex = time.time() - ex

print(ex)
plotter.loglog(x, err)
# p.plot(x, y)
# p.plot(x, exact)
plotter.grid()
plotter.show()
