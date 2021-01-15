import numpy as np
import scipy.stats as st
from phy import Event, Sim, Canal
import matplotlib.pyplot as plt
from utils import pkt
from utils import l_pkt
from utils import animar


class NetDev:
    def __init__(self, bps, len_pkt, delay, nodo):
        self.bps = bps
        self.len_pkt = len_pkt
        self.delay = delay
        self.state = "idle"
        self.nodo = nodo
        self.tol = int(np.ceil(np.log10(bps)))
        self.pkt = self.len_pkt / self.bps
        self.t_delay = 0
        self.pkt_rcv = True

    def set_pkt_rcv(self, var):
        self.pkt_rcv = var

    def start_tx(self, tiempo, S):
        if self.state == "idle":
            if self.pkt_rcv:
                self.t_delay = tiempo
                self.pkt_rcv = False
            self.state = "tx"
            var = tiempo + self.pkt
            S.push(Event(var, "FinishTx", self.nodo), var)  # se programa el final del envio
            S.push(Event(tiempo + self.delay, "StartRx", 0, emisor=self.nodo, l_pkt=self.pkt), tiempo + self.delay)  # se programa la recepcion
            del var
            return True
        return False

    def finish_tx(self):
        self.state = "idle"

    def start_rx(self, tiempo, S, emi=None, len_pkt=None):
        var = tiempo + len_pkt
        S.push(Event(var, "FinishRx", 0, emi, len_pkt), var)
        del var

    def finish_rx(self, C):
        self.state = "idle"
        result = 0
        if C.estado() == 1:
            result = 1  # regresa 1 si la recepcion fue exitosa
        return result  # regresa 0 si hay una colision para contar el numero de paquetes colisionados


def animacion(l, t_f, bps=1e6, t_pkt="const", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()

    tol = int(np.ceil(np.log10(bps)))

    nodos = []

    pkts = []
    inicio = 0
    libre = True

    for x in range(n_nodos):
        len_pkt = l_pkt[t_pkt](bps)
        nodos.append(NetDev(bps, len_pkt, delay, x))

    C.reset()

    while S.size() > 0:
        S.pop()

    t = 0

    var = np.round(np.random.exponential(1 / l), decimals=tol)
    nodo = np.random.randint(1, n_nodos)
    S.push(Event(var, "StartTx", nodo), var)

    while t < t_f:
        e = S.pop()
        t = e.tiempo
        n = e.nodo
        tipo = e.tipo

        if tipo == "StartTx":
            if nodos[n].start_tx(t, S):
                pkts.append(pkt(t, t+nodos[n].pkt, "green", n))
                if libre:
                    inicio = t
                    libre = False
                C.ocupar()
            var = t + np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

        if tipo == "FinishTx":
            nodos[n].finish_tx()

        if tipo == "StartRx":
            nodos[n].start_rx(t, S, e.emisor, e.len_pkt)

        if tipo == "FinishRx":
            nodos[n].finish_rx(C)
            if C.n_trans == 1:
                col = "green"
                if C.col:
                    col = "red"
                pkts.append(pkt(inicio, t, col, n_nodos))
                libre = True
            C.desocupar()
    return pkts


def delay(n_sim, Lambda, bps=1e6, t_pkt="const", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()

    t_delay = []
    cs_delay = []
    ci_delay = []

    tol = int(np.ceil(np.log10(bps)))

    for l in Lambda:
        print(l)
        aux = []
        for con in range(n_sim):
            m = 0
            d = 0
            nodos = []

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(NetDev(bps, len_pkt, delay, x))

            C.reset()

            while S.size() > 0:
                S.pop()

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            while m < 10:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "StartTx":
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    var = t + np.round(np.random.exponential(1 / l), decimals=tol)
                    nodo = np.random.randint(1, n_nodos)
                    S.push(Event(var, "StartTx", nodo), var)

                if tipo == "FinishTx":
                    nodos[n].finish_tx()

                if tipo == "StartRx":
                    nodos[n].start_rx(t, S, e.emisor, e.len_pkt)

                if tipo == "FinishRx":
                    m += nodos[n].finish_rx(C)
                    if C.estado() == 1:
                        d += t - nodos[e.emisor].t_delay
                        nodos[e.emisor].set_pkt_rcv(True)
                    C.desocupar()

            aux.append(d / m)


        t_delay.append(np.mean(aux))
        sup, inf = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        cs_delay.append(sup)
        ci_delay.append(inf)

    return t_delay, cs_delay, ci_delay


def throughput(n_sim, Lambda, bps=1e6, t_pkt="const", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()

    result = []
    ci_sup = []
    ci_inf = []

    tol = int(np.ceil(np.log10(bps)))

    for l in Lambda:
        print(l)
        aux = []
        for con in range(n_sim):
            m = 0
            nodos = []

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(NetDev(bps, len_pkt, delay, x))

            C.reset()

            while S.size() > 0:
                S.pop()

            t = 0

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            while t <= 30:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "StartTx":
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    var = t + np.round(np.random.exponential(1 / l), decimals=tol)
                    nodo = np.random.randint(1, n_nodos)
                    S.push(Event(var, "StartTx", nodo), var)

                if tipo == "FinishTx":
                    nodos[n].finish_tx()

                if tipo == "StartRx":
                    nodos[0].start_rx(t, S, e.emisor, e.len_pkt)

                if tipo == "FinishRx":
                    if nodos[0].finish_rx(C):
                        m += e.len_pkt
                    C.desocupar()

            aux.append(m / t)

        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup

"""
t = 30

val = animacion(1, t)

animar(t, val, "prueba", True)
"""
