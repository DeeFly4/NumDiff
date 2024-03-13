import numpy as np
import functions as f
import matplotlib.pyplot as plotter
import time

N = 999

ex = time.time()

eig, u = f.slv(N, 3)

print(time.time() - ex)

x = np.linspace(0, 1, N + 2)

plotter.plot(x, u[:, 2], label=r'$\sin(\pi x)$')
plotter.plot(x, u[:, 1], label=r'$\sin(2\pi x)$')
plotter.plot(x, u[:, 0], label=r'$\sin(3\pi x)$')
plotter.xlabel(r'$x$')
plotter.ylabel(r'$u(x)$')
plotter.title('First three numerical modes')
plotter.legend()
plotter.grid()
plotter.show()
