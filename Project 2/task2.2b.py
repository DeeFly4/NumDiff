import numpy as np
import functions as f
import math
import time

N = 999


def eq1(x):
	return 700 * (.5 - abs(x - .5))


def eq2(x):
	return 800 * np.sin(math.pi * x) ** 2


def eq3(x):
	return 500 + 500 * np.sin(7 * math.pi * x)


x = np.linspace(0, 1, N + 2)  # Interior grid points and the boundary points

# Evaluate the potential at the interior grid points

v1 = eq1(x[1:-1])
v2 = eq2(x[1:-1])
v3 = eq3(x[1:-1])

ex = time.time()

# I comment out different parts to get different plots

# eig1 = f.schr(N, v1)
# eig2 = f.schr(N, v2)
eig3 = f.schr(N, v3)

print(time.time() - ex)
