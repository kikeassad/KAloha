from skorch.toy import make_regressor
from skorch import NeuralNetRegressor
from sklearn.metrics import mean_squared_error
import torch
from collections import defaultdict
import numpy as np
import scipy.stats as st
from phy import Event, Sim, Canal, State
import matplotlib.pyplot as plt
from kaloha import KNetDev
from utils import l_pkt
from saloha import NetDev
import kaloha
import saloha
import os
import pickle


def prob(l):
    return np.minimum(1.0, 1.0/l)


def throughput_lambda(modelo, channel, n_sim, tam_s, Lambda, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=51):
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
            state = State(tam_s//2)

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(KNetDev(bps, len_pkt, delay, x, slot, tam_s//2))
                nodos[-1].set_prob(p)

            C.reset()

            while S.size() > 0:
                S.pop()

            nodo_tx = []
            t = 0

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            for x in nodos:
                S.push(Event(slot, "FinSlot", x.nodo), slot)

            m = 0
            while t < 250:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "FinSlot":
                    if nodos[n].fin_slot(t, S):
                        if n == 0:
                            for x in nodos:
                                if x in set(nodo_tx):
                                    x.push_state(1)
                                else:
                                    x.push_state(0)
                            nodos_tx.clear()
                            val = C.estado(channel)
                            state.push(val)
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
                            S.push(Event(t + nodos[n].delay + 1/bps, "ACK", x.nodo), t + nodos[n].delay + 1/bps) # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                    C.desocupar()

                if tipo == "ACK":
                    nodos[n].set_AckT(t)
                    if n == 0:
                        for x in nodos:
                            if x in set(nodo_tx):
                                x.push_state(1)
                            else:
                                x.push_state(0)
                        nodos_tx.clear()
                        val = C.estado(channel)
                        state.push(val)
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                        nodo_tx.append(n)
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            while t < 250 + 5 * tam_s:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "FinSlot":
                    if nodos[n].fin_slot(t, S):
                        if n == 0:
                            for x in nodos:
                                if x in set(nodo_tx):
                                    x.push_state(1)
                                else:
                                    x.push_state(0)
                            nodos_tx.clear()
                            val = C.estado(channel)
                            state.push(val)
                        torch.no_grad()
                        estado = np.array(state.get_state() + nodos[n].get_state())
                        estado = estado.astype(np.float32)
                        estado = torch.from_numpy(estado)
                        estado = estado.to(device="cuda")
                        val = modelo.module_(estado).to(device="cpu").detach().numpy()
                        p = prob(val)
                        nodos[n].set_prob(p)
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
                        m += e.len_pkt
                        # enviamos ack
                        for x in nodos:
                            S.push(Event(t + nodos[n].delay + 1/bps, "ACK", x.nodo), t + nodos[n].delay + 1/bps) # Le sumamos 1/bps xq es lo que restamos para que no se coordinen con lso finales de slot
                    C.desocupar()

                if tipo == "ACK":
                    nodos[n].set_AckT(t)
                    if n == 0:
                        for x in nodos:
                            if x in set(nodo_tx):
                                x.push_state(1)
                            else:
                                x.push_state(0)
                        nodos_tx.clear()
                        val = C.estado(channel)
                        state.push(val)
                    if nodos[n].start_tx(t, S):
                        nodo_tx.append(n)
                        C.ocupar()
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            aux.append(m/(t - 250))
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup


def throughput_pktSlot(modelo, channel, n_sim, tam_s, Lambda, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=51):
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
            state = State(tam_s//2)

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(KNetDev(bps, len_pkt, delay, x, slot, tam_s//2))
                nodos[-1].set_prob(p)

            C.reset()

            while S.size() > 0:
                S.pop()

            nodos_tx = []
            t = 0

            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)

            for x in nodos:
                S.push(Event(slot, "FinSlot", x.nodo), slot)

            m = 0
            while t < 250:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "FinSlot":
                    if nodos[n].fin_slot(t, S):
                        if n == 0:
                            for x in nodos:
                                if x in set(nodo_tx):
                                    x.push_state(1)
                                else:
                                    x.push_state(0)
                            nodos_tx.clear()
                            val = C.estado(channel)
                            state.push(val)                            
                            for x in nodos:
                                x.n_pkts = 0
                        if nodos[n].start_tx(t, S):
                            for x in nodos:
                                x.n_pkts += 1
                            C.ocupar()
                            nodos_tx.append(n)

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
                    C.desocupar()

                if tipo == "ACK":
                    nodos[n].set_AckT(t)
                    if n == 0:
                        for x in nodos:
                            if x in set(nodo_tx):
                                x.push_state(1)
                            else:
                                x.push_state(0)
                        nodos_tx.clear()
                        val = C.estado(channel)
                        state.push(val)
                        for x in nodos:
                            x.n_pkts = 0
                        val = C.estado(channel)
                        state.push(val)
                    var = 0
                    if nodos[n].start_tx(t, S):
                        for x in nodos:
                            x.n_pkts += 1
                        C.ocupar()
                        nodos_tx.append(n)
                    S.push(Event(t + slot, "FinSlot", n), t + slot)
            
            while t < 250 + 5 * tam_s:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "FinSlot":
                    if nodos[n].fin_slot(t, S):
                        if n == 0:
                            for x in nodos:
                                if x in set(nodo_tx):
                                    x.push_state(1)
                                else:
                                    x.push_state(0)
                            nodos_tx.clear()
                            val = C.estado(channel)
                            state.push(val)
                        # calculamos la probabilidad
                        torch.no_grad()
                        estado = np.array(state.get_state() + nodos[n].get_state())
                        estado = estado.astype(np.float32)
                        estado = torch.from_numpy(estado)
                        estado = estado.to(device="cuda")
                        val = modelo.module_(estado).to(device="cpu").detach().numpy()
                        p = prob(val)
                        # seteamos la probabilidad
                        nodos[n].set_prob(p)
                        if nodos[n].start_tx(t, S):
                            nodos_tx.append(n)
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
                    if n == 0:
                        for x in nodos:
                            if x in set(nodo_tx):
                                x.push_state(1)
                            else:
                                x.push_state(0)
                        nodos_tx.clear()
                        val = C.estado(channel)
                        state.push(val)
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                        nodos_tx.append(n)
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            aux.append(m/(5 * tam_s))
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup

