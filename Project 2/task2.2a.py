import numpy as np
import functions as f
import time

N = 999

v = np.zeros(N)

ex = time.time()

eig = f.schr(N, v, k=3) # Only want the first 3

print(time.time() - ex)
