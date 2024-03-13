import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plotter


def twopBVP(fvec, a, b, L, N):
	dx = L / (N + 1)
	diagonals = [np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)]
	A = 1 / (dx ** 2) * diags(diagonals, [-1, 0, 1], format="csr") # creating the sparse matrix
	fvec[0] = fvec[0] - a / (dx ** 2)
	fvec[-1] = fvec[-1] - b / (dx ** 2) # fixing the values at beginning and according to equation
	y = linalg.spsolve(A, fvec)
	y = np.insert(y, 0, a)
	y = np.append(y, b) # insert and append the boundary values

	return y


def slv(N, k):
	dx = 1 / (N + 1)
	diagonals = [np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)]
	T = 1 / (dx ** 2) * diags(diagonals, [-1, 0, 1], format="csr")
	eig, u = linalg.eigsh(T, k, which='SM')
	u = np.insert(u, 0, np.zeros([1, k]), axis=0)  # insert zeroes at beginning
	u = np.append(u, np.zeros([1, k]), axis=0)  # append zeroes

	return eig, u


def schr(N, v, k=6):
	dx = 1 / (N + 1)
	diagonals = [np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)]
	T = 1 / (dx ** 2) * diags(diagonals, [-1, 0, 1], format="csr")
	V = diags(v, format="csr") # both matrices in sparse format

	eig, u = linalg.eigsh(V - T, k, which='SM')
	u = np.insert(u, 0, np.zeros([1, k]), axis=0)  # insert zeroes at beginning
	u = np.append(u, np.zeros([1, k]), axis=0)  # append zeroes

	scale = abs(max(eig)) * .2 # scale for plotting purposes.

	x = np.linspace(0, 1, N + 2)

	wave = plotter.figure(1)
	for i in range(k): # plotting in reverse order to get smallest eigenvalues first
		plotter.plot(x, eig[k - 1 - i] + scale * u[:, k - 1 - i], )
	plotter.xlabel(r'$x$')
	plotter.ylabel(r'$\psi$')
	plotter.title('Wave functions', figure=wave)
	plotter.grid()

	dens = plotter.figure(2)
	for i in range(k):
		plotter.plot(x, eig[k - 1 - i] + (scale * abs(u[:, k - 1 - i])) ** 2)

	plotter.xlabel(r'$x$')
	plotter.ylabel(r'$|\psi|^2$')
	plotter.title('Probability densities', figure=dens)
	plotter.grid()
	plotter.show()

	return eig
