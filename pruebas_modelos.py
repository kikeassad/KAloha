import os
import numpy as np
import matplotlib.pyplot as plt
#from entrenar_modelos import crear_modelo
from utils import animar
from kaloha import animacion
"""
np.random.seed(13)
dir = os.listdir("modelos/kaloha")

d = dir[-2]
x = d.rsplit(".")[0].rsplit("_")
print(x)
i_units = int(x[0])
h_layers = int(x[1])
h_units = int(x[2])
channel = x[3] == "True"
reg = crear_modelo(i_units, h_units, h_layers)
reg.initialize()
reg.load_params(f_params="modelos/kaloha/" + d)
print(channel)
t = 30
"""
#val = animacion(1, t, tam_s=i_units, t_pkt="exp", n_nodos=11, model=reg, channel=channel)
#val = animacion(1, t, tam_s=i_units, t_pkt="exp", n_nodos=11)

#animar(t, val, "kaloha", True, n_nodos=11, tam_s=i_units)

# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
