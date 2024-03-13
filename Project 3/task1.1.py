import functions as f
import numpy as np
import time

N = 19
M = 100


def g(x):
	return .5 - np.abs(x - .5)


x = np.linspace(0, 1, N + 2)

gvec = g(x[1:-1])

ex = time.time()

U = f.parTR(gvec, 1, N, M)

f.plotU(U, 1, N, M)

# f.testCFL(g)

print(time.time() - ex)
