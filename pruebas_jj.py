import numpy as np
import scipy.stats as st
from phy import Event, Sim, Canal
from utils import l_pkt
from kaloha import KNetDev
import matplotlib.pyplot as plt
import kaloha
import saloha

np.random.seed(23)

def prob_jj(val, cota):
    if val < 1.6:
        return 1
    return cota


def throughput(n_sim, Lambda, proba, bps=1e6, t_pkt="const", delay=0, n_nodos=51):
    S = Sim()
    C = Canal()
    slot = 1.0

    result = []
    ci_sup = []
    ci_inf = []

    tol = int(np.ceil(np.log10(bps)))
    for l in Lambda:
        print(l)
        aux = []
        for con in range(n_sim):
            nodos = []

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(KNetDev(bps, len_pkt, delay, x, slot))

            C.reset()

            while S.size() > 0:
                S.pop()

            t = 0

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            for x in nodos:
                S.push(Event(slot, "FinSlot", x.nodo), slot)

            m = 0
            while t < 30:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "FinSlot":
                    p = prob_jj(l, proba)
                    nodos[n].set_prob(p)
                    if nodos[n].fin_slot(t, S):
                        if nodos[n].start_tx(t, S):
                            C.ocupar()

                if tipo == "StartTx":
                    nodos[n].set_pkt_snd(True)
                    var = t + np.round(np.random.exponential(1 / l), decimals=tol)
                    nodo = np.random.randint(1, n_nodos)
                    S.push(Event(var, "StartTx", nodo), var)

                if tipo == "FinishTx":
                    nodos[n].finish_tx()

                if tipo == "StartRx":
                    nodos[n].start_rx(t, S, e.emisor, e.len_pkt)

                if tipo == "FinishRx":
                    if nodos[n].finish_rx(C) == 1:
                        m += e.len_pkt
                        # enviamos ack
                        for x in nodos:
                            S.push(Event(t + nodos[n].delay + 1/bps, "ACK", x.nodo), t + nodos[n].delay + 1/bps) # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                    C.desocupar()

                if tipo == "ACK":
                    p = prob_jj(l, proba)
                    nodos[n].set_prob(p)
                    nodos[n].set_AckT(t)
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            aux.append(m/t)
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup


def throughput2(n_sim, Lambda, proba, bps=1e6, t_pkt="const", delay=0, n_nodos=51):
    S = Sim()
    C = Canal()
    slot = 1.0

    result = []
    ci_sup = []
    ci_inf = []

    tol = int(np.ceil(np.log10(bps)))
    for l in Lambda:
        print(l)
        aux = []
        for con in range(n_sim):
            nodos = []
            ack = []

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                ack.append(True)
                nodos.append(KNetDev(bps, len_pkt, delay, x, slot))

            C.reset()

            while S.size() > 0:
                S.pop()

            t = 0

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            for x in nodos:
                S.push(Event(slot, "FinSlot", x.nodo), slot)

            m = 0
            while t < 30:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "init":
                    #if l > 1.6: # Linea pa que salga como en el paper
                    if C.estado(True) == -1:
                        nodos[n].set_prob(proba)
                    if C.estado() == 1:
                        nodos[n].set_prob(1)

                if tipo == "FinSlot":
                    if nodos[n].fin_slot(t, S):
                        S.push(Event(t + 1/bps, "init", n), t + 1/bps)
                        if nodos[n].start_tx(t, S):
                            C.ocupar()

                if tipo == "StartTx":
                    nodos[n].set_pkt_snd(True)
                    var = t + np.round(np.random.exponential(1 / l), decimals=tol)
                    nodo = np.random.randint(1, n_nodos)
                    S.push(Event(var, "StartTx", nodo), var)

                if tipo == "FinishTx":
                    nodos[n].finish_tx()

                if tipo == "StartRx":
                    nodos[n].start_rx(t, S, e.emisor, e.len_pkt)

                if tipo == "FinishRx":
                    if nodos[n].finish_rx(C) == 1:
                        m += e.len_pkt
                        # enviamos ack
                        for x in nodos:
                            S.push(Event(t + nodos[n].delay + 1/bps, "ACK", x.nodo), t + nodos[n].delay + 1/bps) # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                    C.desocupar()

                if tipo == "ACK":
                    nodos[n].set_AckT(t)
                    nodos[n].set_prob(1)
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            aux.append(m/t)
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup


np.random.seed(23)
#Lambda = np.arange(.1, 6.1, .1)
Lambda = [.001, .1, .2, .3, .4, .5, 1, 4/3, 2, 3, 4, 5, 6, 7, 10, 50, 60, 70, 80, 90, 100]
vals = [1, .75, .5, .2]
#vals = [.5, .2, 1]

for x in vals:
    s, ci_inf, ci_sup = throughput2(50, Lambda, x)
    plt.title("Uso del canal")
    plt.ylabel('Uso del canal')
    plt.xlabel('Tasa de arribos')
    plt.grid(True)
    plt.semilogx(Lambda, s)
    plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)

plt.legend(vals, loc='upper left')
plt.show()
