import numpy as np
import scipy.stats as st
from phy import Event, Sim, Canal, State
from utils import pkt
from utils import l_pkt
# Se crean las clases esenciales


class KNetDev:
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
        self.t = 0 #tiempo del ultimo ack
        self.t_temp = 0 #auxialiar para controlar el inicio de los slots
        self.t_delay = 0
        self.pkt_rcv = True
        self.pkt_snd = False

    def set_pkt_snd(self, var):
        self.pkt_snd = var

    def set_pkt_rcv(self, var):
        self.pkt_rcv = var

    def set_AckT(self, t):
        self.t = t

    def push_state(self, x):
        self.s.push(x)

    def get_state(self):
        return self.s.get_state()

    def set_prob(self, p):
        self.p = p

    def start_tx(self, tiempo, S):
        if self.pkt_snd:
            self.pkt_snd = False
            if self.pkt_rcv:
                self.t_delay = tiempo
                self.pkt_rcv = False
            if np.random.rand() <= self.p:
                S.push(Event(tiempo + self.pkt_duration, "FinishTx", self.nodo), tiempo + self.pkt_duration)
                S.push(Event(tiempo + self.delay, "StartRx", 0, emisor=self.nodo, l_pkt=self.pkt_duration), tiempo + self.delay)
                return True
        return False

    def finish_tx(self):
        self.state = "idle"

    def start_rx(self, tiempo, S, emi=None, len_pkt=None):
        var = tiempo + len_pkt - 1/self.bps#le restamos eso para que no se coordine con los finales de slot
        S.push(Event(var, "FinishRx", 0, emi, len_pkt), var)
        del var

    def finish_rx(self, C):
        self.state = "idle"
        if C.estado() == 1:
            return 1  # regresa 1 si la recepcion fue exitosa
        return 0  # regresa 0 si hay una colision para contar el numero de paquetes colisionados

    def fin_slot(self, tiempo, S):
        if self.t == self.t_temp:
            S.push(Event(tiempo + self.slot, "FinSlot", self.nodo), tiempo + self.slot)
            return True
        else:
            self.t_temp = self.t
        return False

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
            nodos.append(KNetDev(bps, len_pkt, delay, x, slot, n_states=i))
            nodos[-1].set_prob(p)

        C.reset()

        while S.size() > 0:
            S.pop()

        nodo_tx = []

        var = np.round(np.random.exponential(1 / l), decimals=tol)
        nodo = np.random.randint(1, n_nodos)
        S.push(Event(var, "StartTx", nodo), var)

        for x in nodos:
            S.push(Event(slot, "FinSlot", x.nodo), slot)

        epsilon = 1 / bps
        S.push(Event(delay + epsilon, "intra_slot", -1), delay + epsilon)

        t = 0
        while t < .75 * tam_s:
            e = S.pop()
            t = e.tiempo
            n = e.nodo
            tipo = e.tipo

            if tipo == "intra_slot":
                S.push(Event(t + slot + delay + epsilon, "intra_slot", -1), t + slot + delay + epsilon)
                for x in nodos:
                    if x.nodo in set(nodo_tx):
                        x.push_state(1)
                    else:
                        x.push_state(0)
                val = C.estado(channel)
                state.push(val)  # True 3 estados en el canal, False 2 estados en el canal
                nodo_tx.clear()

            if tipo == "FinSlot":
                if nodos[n].fin_slot(t, S):
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                        nodo_tx.append(n)

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
                    # enviamos ack
                    for x in nodos:
                        S.push(Event(t + nodos[n].delay + 1 / bps, "ACK", x.nodo), t + nodos[
                            n].delay + 1 / bps)  # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                C.desocupar()

            if tipo == "ACK":
                nodos[n].set_AckT(t)
                if nodos[n].start_tx(t, S):
                    C.ocupar()
                    nodo_tx.append(n)
                S.push(Event(t + slot, "FinSlot", n), t + slot)

        m = 0
        while m < t_datos:
            e = S.pop()
            t = e.tiempo
            n = e.nodo
            tipo = e.tipo

            if tipo == "intra_slot":
                S.push(Event(t + slot + delay + epsilon, "intra_slot", -1), t + slot + delay + epsilon)
                for x in nodos:
                    if x.nodo in set(nodo_tx):
                        x.push_state(1)
                    else:
                        x.push_state(0)
                val = C.estado(channel)
                state.push(val)  # True 3 estados en el canal, False 2 estados en el canal
                nodo_tx.clear()

            if tipo == "FinSlot":
                if nodos[n].fin_slot(t, S):
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                        nodo_tx.append(n)
                        result.append(state.get_state() + nodos[n].get_state() + [l])

            if tipo == "StartTx":
                m += 1
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
                    # enviamos ack
                    for x in nodos:
                        S.push(Event(t + nodos[n].delay + 1 / bps, "ACK", x.nodo), t + nodos[n].delay + 1 / bps)  # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                C.desocupar()

            if tipo == "ACK":
                nodos[n].set_AckT(t)
                if nodos[n].start_tx(t, S):
                    C.ocupar()
                    nodo_tx.append(n)
                    result.append(state.get_state() + nodos[n].get_state() + [l])
                S.push(Event(t + slot, "FinSlot", n), t + slot)

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
    slot = 1.0

    tol = int(np.ceil(np.log10(bps)))

    nodos = []

    pkts = []
    inicio = 0
    libre = True

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

    while t < t_f:
        e = S.pop()
        t = e.tiempo
        n = e.nodo
        tipo = e.tipo

        if tipo == "FinSlot":

            if nodos[n].fin_slot(t, S):
                if nodos[n].start_tx(t, S):
                    pkts.append(pkt(t, t + nodos[n].pkt_duration, "green", n))
                    if libre:
                        inicio = t
                        libre = False
                    C.ocupar()
                if n == 0:
                    pkts.append(pkt(t, t + slot, "black", n, False))

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
                # enviamos ack
                for x in nodos:
                    S.push(Event(t + nodos[n].delay + 1/bps, "ACK", x.nodo), t + nodos[n].delay + 1/bps) # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
            if C.n_trans == 1:
                col = "green"
                if C.col:
                    col = "red"
                pkts.append(pkt(inicio, t, col, n_nodos))
                libre = True
            C.desocupar()

        if tipo == "ACK":
            nodos[n].set_AckT(t)
            if n == 0:
                pkts.append(pkt(t, t + slot, "blue", n))
            if nodos[n].start_tx(t, S):
                pkts.append(pkt(t, t + nodos[n].pkt_duration, "green", n))
                if libre:
                    inicio = t
                    libre = False
                C.ocupar()
            S.push(Event(t + slot, "FinSlot", n), t + slot)

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
                nodos.append(KNetDev(bps, len_pkt, delay, x, slot))
                nodos[-1].set_prob(p)

            C.reset()

            while S.size() > 0:
                S.pop()

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            epsilon = 1 / bps
            S.push(Event(epsilon, "inicio_slot", -1), delay + epsilon)

            for x in nodos:
                S.push(Event(slot, "FinSlot", x.nodo), slot)

            while m < 10:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "FinSlot":
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
                        m += 1
                        d += t - nodos[e.emisor].t_delay
                        nodos[e.emisor].set_pkt_rcv(True)
                        # enviamos ack
                        for x in nodos:
                            S.push(Event(t + nodos[n].delay + 1/bps, "ACK", x.nodo), t + nodos[n].delay + 1/bps) # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                    C.desocupar()

                if tipo == "ACK":
                    nodos[n].set_AckT(t)
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            aux.append(d / m)
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
            nodos = []

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(KNetDev(bps, len_pkt, delay, x, slot))
                nodos[-1].set_prob(p)

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


"""
t = 30

val = animacion(1, t)

animar(t, val, "prueba", True)
"""

