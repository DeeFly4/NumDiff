import functions as f
import numpy as np
import time


def g(x):
	return np.exp(-100 * (x - .5) ** 2)


N = 19
M = 1000
a = 10
tf = 1

x = np.linspace(0, 1, N + 1)

gvec = g(x[:-1])

ex = time.time()

U = f.LaxWensolver(gvec, a, tf, N, M)

f.plotLaxWen(U, N, M, tf)

# l2 = f.LaxWenRMS(U, N, M)
# f.plotRMS(l2, tf, M)

print(time.time() - ex)
print(f"a = {a}")
