import numpy as np
import functions as f
from scipy.linalg import norm
import matplotlib.pyplot as plotter
import time

K = 11
N = np.arange(2, K)
N = np.power(2, N)


def ev(k):  # eigenvalues are -(kpi)^2 for positive integers k
	return -(k * np.pi) ** 2


ev = ev(np.arange(1, 4))  # first 3 exact eigenvalues
err = np.zeros([3, K - 2])  # error array for the 3 eigenvalues, all N

ex = time.time()

for i in range(K - 2):
	eig, u = f.slv(N[i], 3)  # first 3 eigenvalues

	for k in range(3):  # error for k-th eigenvalue
		err[k, i] = norm(eig[k] - ev[2 - k])

print(time.time() - ex)

plotter.loglog(N, err[2, :], label=r'$\lambda_1$')
plotter.loglog(N, err[1, :], label=r'$\lambda_2$')
plotter.loglog(N, err[0, :], label=r'$\lambda_3$')
plotter.xlabel('N')
plotter.ylabel(r'$\lambda_{\Delta x} - \lambda$')
plotter.legend()
plotter.title('Numerical errors')
plotter.grid()
plotter.show()
