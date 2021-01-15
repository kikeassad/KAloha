import numpy as np

lt = 90
N = 22
n = 1
lf = 6 + 3 * (n - 1)
f = 2400 # MHz

d = np.power(10, ((lt + 28 - 20 * np.log10(f) - lf) / N))

print(d)

c1 = 133
c2 = 217

h = np.sqrt(c1*c1 + c2*c2)

print(h/2)