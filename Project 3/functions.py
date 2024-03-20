import numpy as np
from scipy.sparse import diags, linalg, identity
from scipy import meshgrid
import matplotlib.pyplot as plotter
from mpl_toolkits.mplot3d import Axes3D


def generateT(N, L=1):
	dx = L / (N + 1)
	diagonals = [np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)]
	return 1 / (dx ** 2) * diags(diagonals, [-1, 0, 1], format='csr')


def generateST(N):
	dx = 1 / N
	diagonals = [np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)]
	Tdx = diags(diagonals, [-1, 0, 1], format='csr')
	Tdx[-1, 0] = Tdx[0, -1] = 1
	Tdx *= 1 / (dx ** 2)
	Sdx = diags((-np.ones(N - 1), np.ones(N - 1)), [-1, 1], format='csr')
	Sdx[-1, 0] = 1
	Sdx[0, -1] = -1
	Sdx *= 1 / (2 * dx)

	return Sdx, Tdx


def eulerstep(Tdx, uold, dt):
	return uold + dt * Tdx@uold


def parEul(gvec, tf, N, M):
	dt = tf / M
	dx = 1 / (N + 1)
	print(f"CFL = {dt / (dx ** 2)}")
	Tdx = generateT(N)
	U = np.zeros((N, M + 1))
	U[:, 0] = gvec

	for i in range(M):
		U[:, i + 1] = eulerstep(Tdx, U[:, i], dt)

	U = np.vstack((np.zeros(M + 1), U, np.zeros(M + 1)))

	return U


def testCFL(g, tf=1, start=800, N=19):
	i = start
	gvec = g(np.linspace(0, 1, N + 2)[1:-1])
	U = parEul(gvec, tf, N, i)
	while np.any(U > .5) == False:
		i -= 1
		U = parEul(gvec, tf, N, i)

	print('CFL Limit ^')


def TRstep(Tdx, uold, dt):
	I = identity(len(uold))
	A = I - dt / 2 * Tdx
	b = (I + dt / 2 * Tdx)@uold
	return linalg.spsolve(A, b)


def parTR(gvec, tf, N, M):
	dt = tf / M
	dx = 1 / (N + 1)
	print(f"CFL = {dt / (dx ** 2)}")
	Tdx = generateT(N)
	U = np.zeros((N, M + 1))
	U[:, 0] = gvec

	for i in range(M):
		U[:, i + 1] = TRstep(Tdx, U[:, i], dt)

	U = np.vstack((np.zeros(M + 1), U, np.zeros(M + 1)))

	return U


def plotU(U, tf, N, M):
	xx = np.linspace(0, 1, N + 2)
	tt = np.linspace(0, tf, M + 1)
	T, X = meshgrid(tt, xx)

	ax = plotter.axes(projection='3d')
	ax.plot_surface(T, X, U, cmap='viridis')
	ax.set_title('Diffusion')
	ax.set_xlabel(r'$t$')
	ax.set_ylabel(r'$x$')
	ax.set_zlabel(r'$u(x,\, t)$')
	plotter.show()


def LaxWenstep(u, amu):
	unew = np.zeros(len(u))

	for i in range(len(u) - 1):
		unew[i] = amu / 2 * (1 + amu) * u[i - 1] + (1 - amu ** 2) * u[i] - amu / 2 * (1 - amu) * u[i + 1]

	unew[-1] = amu / 2 * (1 + amu) * u[-2] + (1 - amu ** 2) * u[-1] - amu / 2 * (1 - amu) * u[0]

	return unew


def LaxWensolver(gvec, a, tf, N, M):
	dt = tf / M
	dx = 1 / N
	print(f"CFL = {dt / dx}")
	amu = a * dt / dx
	print(f"amu = {amu}")
	U = np.zeros((N, M + 1))
	U[:, 0] = gvec

	for i in range(M):
		U[:, i + 1] = LaxWenstep(U[:, i], amu)

	U = np.vstack((U, U[0, :]))

	return U


def LaxWenRMS(U, N, M):
	l2 = np.zeros(M + 1)
	for i in range(M + 1):
		l2[i] = np.linalg.norm(U[:-1, i])
	return l2 * 1 / np.sqrt(N)


def plotLaxWen(U, N, M, tf):
	xx = np.linspace(0, 1, N + 1)
	tt = np.linspace(0, tf, M + 1)
	T, X = meshgrid(tt, xx)

	ax = plotter.axes(projection='3d')
	ax.plot_surface(T, X, U, cmap='viridis')
	ax.set_title('Advection, a = -0.5')
	ax.set_xlabel(r'$t$')
	ax.set_ylabel(r'$x$')
	ax.set_zlabel(r'$u(x,\, t)$')
	plotter.show()


def plotRMS(l2, tf, M):
	t = np.linspace(0, tf, M + 1)
	plotter.plot(t, l2)
	plotter.title('RMS norm')
	plotter.xlabel(r'$t$')
	plotter.ylabel(r'RMS($t$)')
	plotter.show()


def convdifstep(u, a, d, dt):
	Sdx, Tdx = generateST(len(u))
	return TRstep(d * Tdx - a * Sdx, u, dt)


def convdifsolver(gvec, a, d, N, M, tf=1):
	dt = tf / M
	dx = 1 / N
	print(f"Pe = {abs(a / d)}")
	print(f"MeshPe = {abs(a / d) * dx}")
	U = np.zeros((N, M + 1))
	U[:, 0] = gvec

	for i in range(M):
		U[:, i + 1] = convdifstep(U[:, i], a, d, dt)

	U = np.vstack((U, U[0, :]))

	return U


def plotConvDif(U, N, M, tf=1):
	xx = np.linspace(0, 1, N + 1)
	tt = np.linspace(0, tf, M + 1)
	T, X = meshgrid(tt, xx)

	ax = plotter.axes(projection='3d')
	ax.plot_surface(T, X, U, cmap='viridis')
	ax.set_title(r'Viscous Burgers, $d=0.005$')
	ax.set_xlabel(r'$t$')
	ax.set_ylabel(r'$x$')
	ax.set_zlabel(r'$u(x,\, t)$')
	plotter.show()


def LW(u, dx, dt):
	unew = np.zeros(len(u))

	for i in range(len(u) - 1):
		unew[i] = u[i] - dt * u[i] * (u[i + 1] - u[i - 1]) / (2 * dx)
		unew[i] += dt ** 2 / 2 * (2 * u[i] * ((u[i + 1] - u[i - 1]) / (2 * dx)) ** 2)
		unew[i] += dt ** 2 / 2 * u[i] ** 2 *\
				   (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2)

	unew[-1] = u[-1] - dt * u[-1] * (u[0] - u[-2]) / (2 * dx)
	unew[-1] += dt ** 2 / 2 * (2 * u[-1] * ((u[0] - u[-2]) / (2 * dx)) ** 2)
	unew[-1] += dt ** 2 / 2 * u[-1] ** 2 * (u[0] - 2 * u[-1] + u[-2]) / (dx ** 2)

	return unew


def burgersstep(Tdx, u, d, dx, dt):
	I = identity(len(u))
	A = I - d * dt / 2 * Tdx
	b = LW(u, dx, dt) + d * dt / 2 * Tdx@u
	return linalg.spsolve(A, b)


def burgerssolver(gvec, d, N, M, tf=1):
	Tdx = generateST(N)[1]
	dt = tf / M
	dx = 1 / N
	U = np.zeros((N, M + 1))
	U[:, 0] = gvec

	for i in range(M):
		U[:, i + 1] = burgersstep(Tdx, U[:, i], d, dx, dt)

	U = np.vstack((U, U[0, :]))

	return U
