from kaloha import throughput2
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(31)
Lambda = np.arange(.1, 6.1, .1)
#Lambda = [.001, .1, .2, .3, .4, .5, 1, 4/3, 2, 3, 4, 5, 6, 7, 10, 50, 60, 70, 80, 90, 100]
#vals = [1, .75, .5, .2]
#vals = [.5, .2, 1]

#for x in vals:
s, ci_inf, ci_sup = throughput2(200, Lambda)
plt.title("Uso del canal")
plt.ylabel('Uso del canal')
plt.xlabel('Tasa de arribos')
plt.grid(True)
plt.plot(Lambda, s)
plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)

#plt.legend(vals, loc='upper left')
plt.show()
