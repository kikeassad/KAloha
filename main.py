import kaloha
import aloha
import saloha
from utils import animar
import matplotlib.pyplot as plt
import numpy as np
# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    """
    t = 30
    val = kaloha.animacion(1, t, t_pkt="exp")
    animar(t, val, "kaloha_exp", True)
    """
    Lambda = np.arange(.1, 6.1, .1)
    s, ci_inf, ci_sup = kaloha.throughput(100, Lambda)
    plt.title("Uso del canal")
    plt.ylabel('Uso del canal')
    plt.xlabel('Tasa de arribos')
    plt.grid(True)
    # plt.stackplot(Lambda, tp, s, labels=[d.rsplit(".")[0], "Kaloha p=1"])
    plt.plot(Lambda, s)
    #plt.legend(loc='upper right', labels=[d.rsplit(".")[0], "Kaloha p=1"])
    plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)
    plt.savefig("img/tp/kaloha")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
