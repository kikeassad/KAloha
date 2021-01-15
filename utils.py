import matplotlib.patches as patches
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
""" 
# filepaths
fp_in = "img/*.png"
fp_out = "img/kaloha.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1000, loop=0)
"""


class pkt:
    def __init__(self, inicio, fin, color, nodo, fill=True):
        self.left = inicio
        self.width = fin - inicio
        self.bottom = nodo
        self.height = .9
        self.color = color
        self.fill = fill

    def draw(self):
        p = patches.Rectangle((self.left, self.bottom), self.width, self.height, fill=self.fill, fc=self.color)
        return p

def constante(num):
    return num


def exp(num):
    val = np.random.exponential(num)
    if val > num:
        val = num
    return val


def uniforme(num):
    return np.random.randint(num/2, num)


l_pkt = {
    "uniforme": uniforme,
    "exp": exp,
    "const": constante
}


def animar(t_f, pkts, name, gif, n_nodos=11):
    y_names = []
    for x in range(n_nodos + 1):
        y_names.append("nodo " + str(x))
    y_names.append("Canal")
    """ """
    count = 0
    for i in np.linspace(5, t_f, num=60):
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(i - 5, i)
        ax.set_ylim(0, 12)
        for x in pkts:
            p = x.draw()
            ax.add_patch(p)
        plt.plot([], [])
        plt.yticks(np.arange(13), y_names)
        plt.grid(True)
        plt.draw()

        if gif:
            var = ""
            if count < 10:
                var = "00" + str(count)
            else:
                var = "0" + str(count)
            plt.savefig("img/" + var + ".png")
            count += 1
        else:
            plt.pause(.1)
    if gif:
        fp_in = "img/*.png"
        fp_out = "img/" + name + ".gif"

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=1000, loop=0)

