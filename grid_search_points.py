import numpy as np

for a in np.arange(1, 2, 0.01):
    for b in np.arange(1, 2**0.5, 0.01):
        res = a * ( b ** 2 )
        if abs(res - 2) < 0.001:
            print("Depth: %.2f, Width: %.2f" % (a,b))