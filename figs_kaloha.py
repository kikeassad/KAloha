import matplotlib.pyplot as plt
import numpy as np


def fig8(a, l , T=1):
    return a*l*T*np.exp(-a*l*T)/(1+l*T*(a*np.exp(-a*l*T)-np.exp(-l*T)))


Lambda = [.001, .1, .2, .3, .4, .5, 1, 4/3, 2, 3, 4, 5, 6, 7, 10, 50, 60, 70, 80, 90, 100]
vals = [1, .75, .5, .2]

for a in vals:
    x = []
    for l in Lambda:
        x.append(fig8(a,l))
    plt.semilogx(Lambda, x)
plt.grid(True)
plt.legend(vals, loc='upper left')
plt.show()
