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


def crear_modelo(i_units, h_units, h_layers):
    model = make_regressor(input_units=i_units, hidden_units=h_units, num_hidden=h_layers)

    regressor = NeuralNetRegressor(
        model,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=.99,
        max_epochs=100,
        verbose=0,
        lr=0.01,
        device="cuda",
        train_split=None,
        warm_start=True,
    )
    return regressor


def throughput_kaloha(modelo, channel, n_sim, tam_s, Lambda, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()
    slot = 1.0

    result = []
    ci_sup = []
    ci_inf = []

    error = []
    e_sup = []
    e_inf = []

    tol = int(np.ceil(np.log10(bps)))
    for l in Lambda:
        print(l)
        aux = []
        aux1 = []
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

            epsilon = 1 / bps
            S.push(Event(delay + epsilon, "intra_slot", -1), delay + epsilon)

            m = 0
            err = 0
            pkts = 0
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
                    if nodos[n].start_tx(t, S):
                        C.ocupar()
                    S.push(Event(t + slot, "FinSlot", n), t + slot)

            while t < 30 + 1.5 * tam_s:
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
                        torch.no_grad()
                        estado = np.array(state.get_state() + nodos[n].get_state())
                        estado = estado.astype(np.float32)
                        estado = torch.from_numpy(estado)
                        estado = estado.to(device="cuda")
                        val = modelo.module_(estado).to(device="cpu").detach().numpy()
                        p = prob(val)
                        pkts += 1
                        err += np.abs(val - l)# error absoluto
                        nodos[n].set_prob(p)
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

            aux.append(m/(30 + 1.5 * tam_s))
            aux1.append(err/pkts)
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

        error.append(np.mean(aux1))
        inf, sup = st.t.interval(.95, len(aux1) - 1, loc=np.mean(aux1), scale=st.sem(aux1))
        e_sup.append(sup[0])
        e_inf.append(inf[0])

    return result, ci_inf, ci_sup, error, e_inf, e_sup


np.random.seed(23)
#entrenar_modelos(kaloha, "modelos/kaloha/")
dir = os.listdir("modelos/kaloha")
Lambda = np.arange(.1, 6.1, .1)
tp, ci_i, ci_s = kaloha.throughput(200, Lambda, t_pkt="exp")

pickle.dump([tp, ci_i, ci_s], open("vars/kaloha/ref.pkl", "wb"))

for d in dir:
    x = d.rsplit(".")[0].rsplit("_")
    print(x)
    i_units = int(x[0])
    h_layers = int(x[1])
    h_units = int(x[2])
    channel = x[3] == "True"

    reg = crear_modelo(i_units, h_units, h_layers)
    reg.initialize()
    reg.load_params(f_params="modelos/kaloha/" + d)

    s, ci_inf, ci_sup, e, e_inf, e_sup = throughput_kaloha(reg, channel, 200, i_units, Lambda)
    """ """
    fig, ax1 = plt.subplots(2)
    ax1[0].set_title("Uso del canal")
    ax1[0].set_ylabel('Uso del canal')
    ax1[0].set_xlabel('Tasa de arribos')
    ax1[0].grid(True)
    #plt.stackplot(Lambda, tp, s, labels=[d.rsplit(".")[0], "Kaloha p=1"])
    ax1[0].plot(Lambda, s, Lambda, tp)
    ax1[0].legend(loc='upper right', labels=[d.rsplit(".")[0], "Kaloha p=1"])
    ax1[0].fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)
    ax1[0].fill_between(Lambda, ci_i, ci_s, color='r', alpha=.1)

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-
    ax1[1].plot(Lambda, e, color="y")
    ax1[1].fill_between(Lambda, e_inf, e_sup, color='y', alpha=.1)

    #fig.tight_layout()

    #plt.show()
    pickle.dump([s, ci_inf, ci_sup, e, e_inf, e_sup], open("vars/kaloha/"+str(i_units) + "_" + str(h_layers) + "_" + str(h_units) + "_" + str(channel) +"ref"".pkl", "wb"))
    plt.savefig("img/kaloha/" + d.rsplit(".")[0] + ".png")
    plt.clf()
