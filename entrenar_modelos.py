from skorch.toy import make_regressor
from skorch import NeuralNetRegressor
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
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

conv = defaultdict(lambda: 1)
conv["b"] = 1
conv["kb"] = 1024
conv["mb"] = 1048576
conv["gb"] = 1073741824

tam_palabra = [16, 32, 64, 128, 256, 512, 1024]


def mem_2_param(val, pre, b):
    """
    val valor
    pre es el prefijo, puede ser b, kb, mb o gb
    b indica si esta en bits(True) o bytes(false)
    """
    x = 8
    if b:
        x = 1
    tam = val * x * conv[pre]
    return tam // 64

#params = [mem_2_param(1, "mb", False)]
params = [mem_2_param(10, "kb", False), mem_2_param(100, "kb", False), mem_2_param(1, "mb", False), mem_2_param(10, "mb", False), mem_2_param(100, "mb", False)]
#usar metodología para la buscade de a topología


def arq(val, i):
    """
    val es el número total de parametros
    i es el número de entradas a la red
    """
    result = []
    if np.round(val / (i + 1)) >= i:
        result.append((int(np.round(val / (i + 1))), 1))
    a_1 = i
    a_2 = i
    k = 2

    while a_1 >= i or a_2 >= i:
        a_1 = np.round((-(i + 1) + np.sqrt((i + 1) ** 2 - 4 * (-1 * val) * (k - 1))) / (2 * (k - 1)))
        a_2 = np.round((-(i + 1) - np.sqrt((i + 1) ** 2 - 4 * (-1 * val) * (k - 1))) / (2 * (k - 1)))
        if a_1 >= i:
            if np.abs(i * a_1 + (k - 1) * a_1 * a_1 - val) < 10:
                if k < a_1:
                    result.append((int(a_1) - 1, k)) #restamos 1 por la neurona del bias
                else:
                    result.append((k - 1, int(a_1)))

        if a_2 >= i:
            if np.abs(i * a_2 + (k - 1) * a_2 * a_2 - val) < 10:
                if k < a_2:
                    result.append((int(a_2) - 1, k))
                else:
                    result.append((k - 1, int(a_2)))

        k += 1
    return result


def prob(l):
    return np.minimum(1.0, 1.0/l)


def throughput_kaloha(modelo, channel, n_sim, tam_s, Lambda, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=11):
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

            epsilon = 1 / bps
            S.push(Event(delay + epsilon, "intra_slot", -1), delay + epsilon)

            m = 0
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
                        p = prob(modelo.module_(estado).to(device="cpu").detach().numpy())
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

            aux.append(m/30.0)
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup


def throughput_saloha(modelo, channel, n_sim, tam_s, Lambda, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=11):
    S = Sim()
    C = Canal()
    slot = 1.0

    result = []
    ci_sup = []
    ci_inf = []

    tol = int(np.ceil(np.log10(bps)))

    i = tam_s // 2

    for l in Lambda:
        print(l)
        aux = []
        for con in range(n_sim):
            m = 0
            nodos = []

            for x in range(n_nodos):
                len_pkt = l_pkt[t_pkt](bps)
                nodos.append(NetDev(bps, len_pkt, delay, x, slot, n_states=i))
                nodos[-1].set_prob(p)

            C.reset()

            while S.size() > 0:
                S.pop()

            t = 0

            nodo_tx = []

            state = State(i)

            epsilon = 1 / bps
            var = np.round(np.random.exponential(1 / l), decimals=tol)
            nodo = np.random.randint(1, n_nodos)
            S.push(Event(var, "StartTx", nodo), var)
            S.push(Event(slot - epsilon, "fin_slot", -1), slot - epsilon)
            S.push(Event(epsilon, "inicio_slot", -1), epsilon)

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

            while t < 1.5 * tam_s + 30:
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

                    torch.no_grad()
                    estado = np.array(state.get_state() + nodos[n].get_state())

                    estado = estado.astype(np.float32)
                    estado = torch.from_numpy(estado)
                    estado = estado.to(device="cuda")
                    p = prob(modelo.module_(estado).to(device="cpu").detach().numpy())
                    nodos[n].set_prob(p)

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
                    m += nodos[n].finish_rx(C) * e.len_pkt
                    C.desocupar()

            aux.append(m/30.0)

        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup


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


def entrenar_modelos(ambiente, src="modelos/"):
    result = []
    col = [True, False]
    counter = 0
    for p in params:
        for word in tam_palabra:
            topologias = arq(p, word)
            if len(topologias) > 0:
                for top in topologias:
                    i_units = word
                    h_units = top[0]
                    h_layers = top[1]
                    for i in col:
                        reg = crear_modelo(i_units, h_units, h_layers)
                        l_data = 0
                        """Entrenamiento"""
                        while l_data < 10 * p:
                            data = ambiente.crear_dataset(10000, channel=i, tam_s=word)
                            Y = data[:, -1]
                            Y = Y.reshape(-1, 1)
                            Y = Y.astype(np.float32)
                            X = data[:, :-1]
                            X = X.astype(np.float32)
                            reg.fit(X, Y)
                            del data
                            l_data += 10000

                        reg.save_params(
                            f_params=src + str(i_units) + "_" + str(h_layers) + "_" + str(h_units) + "_" + str(
                                i) + '.pkl')

                        E_in = mean_squared_error(reg.predict(X), Y)

                        """Validación"""
                        data = ambiente.crear_dataset(1000, channel=i, tam_s=word)

                        Y = data[:, -1]
                        Y = Y.reshape(-1, 1)
                        Y = Y.astype(np.float32)
                        X = data[:, :-1]
                        X = X.astype(np.float32)

                        E_out = mean_squared_error(reg.predict(X), Y)

                        result.append([str(i_units) + "_" + str(h_layers) + "_" + str(h_units) + "_" + str(i), E_in, E_out])
                        print(str(i_units) + "_" + str(h_layers) + "_" + str(h_units) + "_" + str(i), counter)
                        """
                        Lambda = np.arange(.1, 6.1, .1)
                        s, ci_inf, ci_sup = throughput_kaloha(reg, i, 200, word, Lambda)
                        plt.title("Uso del canal")
                        plt.ylabel('Uso del canal')
                        plt.xlabel('Tasa de arribos')
                        plt.legend(['Uso promedio del canal'], loc='upper right')
                        plt.grid(True)
                        plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)
                        plt.plot(Lambda, s)
                        plt.savefig()
                        """
                        del reg
                        del data
                        del X
                        del Y
                        counter += 1
    print(result)

    np.savetxt("results.csv", result, delimiter=",", fmt='%s')


entrenar_modelos(kaloha, "modelos/kaloha/")
dir = os.listdir("modelos/kaloha")
Lambda = np.arange(.2, 6.2, .2)
tp, ci_i, ci_s = kaloha.throughput(100, Lambda, t_pkt="exp")
"""
plt.stackplot(Lambda, tp)
plt.grid(True)
plt.show()
"""
var = []
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

    s, ci_inf, ci_sup = throughput_kaloha(reg, channel, 100, i_units, Lambda)
    #s, ci_inf, ci_sup = throughput_saloha(reg, channel, 100, i_units, Lambda)
    print(len(s))
    """ """
    plt.title("Uso del canal")
    plt.ylabel('Uso del canal')
    plt.xlabel('Tasa de arribos')
    plt.grid(True)
    #plt.stackplot(Lambda, tp, s, labels=[d.rsplit(".")[0], "Kaloha p=1"])
    plt.plot(Lambda, s, Lambda, tp)
    plt.legend(loc='upper right', labels=[d.rsplit(".")[0], "Kaloha p=1"])
    plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)
    plt.fill_between(Lambda, ci_i, ci_s, color='r', alpha=.1)

    plt.savefig("img/kaloha/" + d.rsplit(".")[0] + ".png")
    plt.clf()
    var.append((d.rsplit(".")[0], tp))

with open('data.pk', 'wb') as filehandle:
    pickle.dump(var, filehandle)


#reg = torch.load("modelos/kaloha/" + dir[0])

#print(reg)
