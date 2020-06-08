import numpy as np

for a in np.arange(1, 2, 0.1):
    for b in np.arange(1, 2**0.5, 0.1):
        res = a * ( b ** 2 )
        if abs(res - 2) < 0.1:
            print("Depth: %.2f, Width: %.2f" % (a,b))