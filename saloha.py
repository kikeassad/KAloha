import numpy as np
import scipy.stats as st
from phy import Event, Sim, Canal, State
import matplotlib.pyplot as plt
from utils import pkt
from utils import l_pkt
from utils import animar


class NetDev:
    def __init__(self, bps, len_pkt, delay, nodo, slot, n_states=10):
        self.bps = bps
        self.len_pkt = len_pkt
        self.pkt_duration = len_pkt / bps
        self.delay = delay
        self.state = "idle"
        self.nodo = nodo
        self.slot = slot
        self.p = 1
        self.s = State(n_states)
        self.tol = int(np.ceil(np.log10(bps)))
        self.epsilon = 1 / bps
        self.t_delay = 0
        self.pkt_rcv = True

    def set_pkt_rcv(self, var):
        self.pkt_rcv = var

    def push_state(self, x):
        self.s.push(x)

    def get_state(self):
        return self.s.get_state()

    def set_prob(self, p):
        self.p = p

    def start_tx(self, tiempo, S):
        t = int(np.floor(tiempo * 10 ** self.tol))
        ranura = int(np.floor(self.slot * 10 ** self.tol))
        if self.state == "idle":
            if t % ranura == 0:
                x = np.random.rand()
                if x < self.p:
                    if self.pkt_rcv:
                        self.t_delay = tiempo
                        self.pkt_rcv = False
                    self.state = "tx"
                    var = tiempo + self.pkt_duration
                    S.push(Event(var - self.epsilon, "FinishTx", self.nodo), var - self.epsilon)  # se programa el final del envio
                    S.push(Event(tiempo + self.delay, "StartRx", 0, emisor=self.nodo, l_pkt=self.pkt_duration), tiempo + self.delay)  # se programa la recepcion
                    del var
                    return True
                else:
                    return False
        var = (t // ranura + 1) * ranura / np.floor(self.slot * 10 ** self.tol)
        S.push(Event(var, "RSTx", self.nodo), var)
        del var
        return False

    def finish_tx(self):
        self.state = "idle"

    def start_rx(self, tiempo, S, emi=None, len_pkt=None):
        if self.state == "idle":
            self.state = "rx"
        var = tiempo + len_pkt - self.epsilon  # le restamos eso para que no se coordine con los finales de slot
        S.push(Event(var, "FinishRx", 0, emi, len_pkt), var)
        del var

    def finish_rx(self, C):
        self.state = "idle"
        result = 0
        if C.estado() == 1:
            result = 1  # regresa 1 si la recepcion fue exitosa
        return result  # regresa 0 si hay una colision para contar el numero de paquetes colisionados

# Creamos las funciones que nos permitiran crear las simulaciones


def crear_dataset(n_datos, channel=False, bps=1e6, t_pkt="exp", delay=0, n_nodos=11, tam_s=8, p=1):
    S = Sim()
    C = Canal()
    Lambda = np.arange(0.01, 10.01, .01)
    slot = 1.0

    result = []

    tol = int(np.ceil(np.log10(bps)))

    i = tam_s // 2

    t_datos = 10

    if n_datos // 1000 > 10:
        t_datos = n_datos // 1000

    for l in Lambda:

        nodos = []
        state = State(i)

        for x in range(n_nodos):
            len_pkt = l_pkt[t_pkt](bps)
            nodos.append(NetDev(bps, len_pkt, delay, x, slot, n_states=i))
            nodos[-1].set_prob(p)

        C.reset()

        while S.size() > 0:
            S.pop()

        nodo_tx = []

        epsilon = 1 / bps
        var = np.round(np.random.exponential(1 / l), decimals=tol)
        nodo = np.random.randint(1, n_nodos)
        S.push(Event(var, "StartTx", nodo), var)
        S.push(Event(slot - epsilon, "fin_slot", -1), slot - epsilon)
        S.push(Event(epsilon, "inicio_slot", -1), epsilon)

        t = 0
        while t < .75 * tam_s:
            e = S.pop()
            t = e.tiempo
            n = e.nodo
            tipo = e.tipo

            if tipo == "inicio_slot":
                S.push(Event(t + slot, "inicio_slot", -1), t + slot)
                for x in nodos:
                    if x.nodo in set(nodo_tx):
                        x.push_state(1)
                    else:
                        x.push_state(0)
                val = C.estado(channel)
                state.push(val)  # True 3 estados en el canal, False 2 estados en el canal

            if tipo == "fin_slot":
                S.push(Event(t + slot, "fin_slot", -1), t + slot)
                nodo_tx.clear()

            if tipo == "RSTx":
                if nodos[n].start_tx(t, S):
                    nodo_tx.append(n)
                    C.ocupar()

            if tipo == "StartTx":
                if nodos[n].start_tx(t, S):
                    nodo_tx.append(n)
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
                C.desocupar()

        m = 0
        while m < t_datos:
            e = S.pop()
            t = e.tiempo
            n = e.nodo
            tipo = e.tipo

            if tipo == "inicio_slot":
                S.push(Event(t + slot, "inicio_slot", -1), t + slot)
                for x in nodos:
                    if x.nodo in set(nodo_tx):
                        x.push_state(1)
                    else:
                        x.push_state(0)
                val = C.estado(channel)
                state.push(val)  # True 3 estados en el canal, False 2 estados en el canal

            if tipo == "fin_slot":
                S.push(Event(t + slot, "fin_slot", -1), t + slot)
                nodo_tx.clear()

            if tipo == "RSTx":
                if nodos[n].start_tx(t, S):
                    nodo_tx.append(n)
                    C.ocupar()
                    result.append(state.get_state() + nodos[n].get_state() + [l])

            if tipo == "StartTx":
                m += 1
                if nodos[n].start_tx(t, S):
                    nodo_tx.append(n)
                    C.ocupar()
                    result.append(state.get_state() + nodos[n].get_state() + [l])
                var = t + np.round(np.random.exponential(1 / l), decimals=tol)
                nodo = np.random.randint(1, n_nodos)
                S.push(Event(var, "StartTx", nodo), var)

            if tipo == "FinishTx":
                nodos[n].finish_tx()

            if tipo == "StartRx":
                nodos[n].start_rx(t, S, e.emisor, e.len_pkt)

            if tipo == "FinishRx":
                nodos[n].finish_rx(C)
                C.desocupar()

    result = np.array(result)
    np.random.shuffle(result)
    var = result[:n_datos]

    del result
    del S
    del C
    del Lambda

    return var


def animacion(l, t_f, bps=1e6, t_pkt="const", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()
    slot = 1

    tol = int(np.ceil(np.log10(bps)))

    nodos = []

    pkts = []
    inicio = 0
    libre = True

    for x in range(n_nodos):
        len_pkt = l_pkt[t_pkt](bps)
        nodos.append(NetDev(bps, len_pkt, delay, x, slot))

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

        if tipo == "RSTx":
            if nodos[n].start_tx(t, S):
                pkts.append(pkt(t, t + nodos[n].pkt_duration, "green", n))
                if libre:
                    inicio = t
                    libre = False
                C.ocupar()

        if tipo == "StartTx":
            if nodos[n].start_tx(t, S):
                pkts.append(pkt(t, t+nodos[n].pkt_duration, "green", n))
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


def delay(n_sim, Lambda, p=1, bps=1e6, t_pkt="const", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()
    slot = 1.0

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
                nodos.append(NetDev(bps, len_pkt, delay, x, slot))
                nodos[-1].set_prob(p)

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

                if tipo == "RSTx":
                    if nodos[n].start_tx(t, S):
                        C.ocupar()

                if tipo == "StartTx":
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    var = t + np.round(np.random.exponential(1 / l), decimals=tol)
                    nodo = np.random.randint(1, n_nodos)
                    S.push(Event(var, "StartTx", nodo), var)

                if tipo == "FinishTx":
                    nodos[n].finish_tx()

                if tipo == "StartRx":
                    nodos[n_nodos - 1].start_rx(t, S, e.emisor, e.len_pkt)

                if tipo == "FinishRx":
                    if nodos[n_nodos - 1].finish_rx(C) == 1:
                        m += 1
                        d += t - nodos[e.emisor].t_delay
                        nodos[e.emisor].set_pkt_rcv(True)
                    C.desocupar()

            aux.append(d/m)

        t_delay.append(np.mean(aux))
        sup, inf = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        cs_delay.append(sup)
        ci_delay.append(inf)

    return t_delay, cs_delay, ci_delay


def throughput(n_sim, Lambda, p=1, bps=1e6, t_pkt="const", delay=0, n_nodos=11):
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
            m = 0
            nodos = []

            len_pkt = l_pkt[t_pkt](bps)

            for x in range(n_nodos):
                nodos.append(NetDev(bps, len_pkt, delay, x, slot))
                nodos[x].set_prob(p)

            C.reset()

            while S.size() > 0:
                S.pop()

            t = 0

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            while t < 50:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "RSTx":
                    if nodos[n].start_tx(t, S):
                        C.ocupar()

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
                    if nodos[n].finish_rx(C):
                        m += e.len_pkt
                    C.desocupar()

            aux.append(m/t)

        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup


#p = animacion(50, 30)
#animar(30, p, "algo", False)