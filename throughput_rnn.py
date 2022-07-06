from kaloha import crear_dataset_pktSlot
from kaloha import crear_dataset_l
from kaloha import throughput
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from phy import Event, Sim, Canal, State
import matplotlib.pyplot as plt
from kaloha import KNetDev
from utils import l_pkt
from saloha import NetDev
import os

torch.manual_seed(13)
np.random.seed(17)

def prob(l):
    if l == 0:
        return 1
    return np.minimum(1.0, 1.0/l)

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers)  
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(1)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


def crear_entrada(input_data):
    x = []
    y = input_data[:, -1]
    ins = input_data[:,:-1]

    for i in ins:
        aux = []
        var = len(i)//2
        for j in range(var):
            aux.append([float(i[j]), float(i[var + j])])
        x.append(aux)


    return np.array(x, dtype=float), np.array(y, dtype=float)


def predict(modelo, entrada, dev):
    modelo.eval()
    valor = torch.FloatTensor(entrada)
    valor = torch.unsqueeze(valor, 0)
    valor = valor.to(device)
    output, hidden = modelo(valor)
    output = output.to("cpu").detach().numpy()

    return np.squeeze(output)[-1]


def throughput_kaloha( channel, n_sim, tam_s, Lambda, modelo=None, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=61):
    S = Sim()
    C = Canal()
    slot = 1.0
    t_lim = 100

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

            while t < t_lim:
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

            m = 0

            while t < 100 + t_lim:
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
                        estado = np.array(state.get_state() + nodos[n].get_state())
                        var = tam_s//2
                        seq = []
                        for i in range(var):
                            seq.append([float(estado[i]), float(estado[var + i])])
                        """ """
                        seq = np.array(seq, dtype=float)
                        if modelo != None:
                            val = predict(modelo, seq, device)
                            p = prob(val)
                        else:
                            p = 1
                            #if nodos[n].n_pkts > 0:
                            #    p = 1/nodos[n].n_pkts
                        #print("valor p: "+ str(p))
                        #print("valor pkt en canal: "+ str(nodos[n].n_pkts))
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

            aux.append(m/(100))
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup

device = torch.device("cuda")

Lambda = np.arange(0.1, 6.0, .1)
s, ci_inf, ci_sup = throughput(75, Lambda)# prob = 1

s_, ci_inf_, ci_sup_ = throughput_kaloha(True, 10, 8, Lambda)

plt.title("Uso del canal")
plt.ylabel('Uso del canal')
plt.xlabel('Tasa de arribos')
plt.grid(True)

plt.plot(Lambda, s)
plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)

plt.plot(Lambda, s_)
plt.fill_between(Lambda, ci_inf_, ci_sup_, color='r', alpha=.1)
plt.legend(loc='upper right', labels=["Kaloha p=1", "Kaloha optimo"])

plt.show()

"""
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

dir = os.listdir("modelos/rnn")
#model.load_state_dict(torch.load("modelos/rnn.mod"))

Lambda = np.arange(0.1, 6.0, .1)
s, ci_inf, ci_sup = throughput(75, Lambda)# prob = 1

for d in dir:
    x = d.rsplit(".")[0].rsplit("-")
    model = Model(input_size=2, output_size=1, hidden_dim=int(x[1]), n_layers=1)
    model.load_state_dict(torch.load("modelos/rnn/"+d))
    model = model.to(device)
    
    i_units = int(x[0])

    s_, ci_inf_, ci_sup_ = throughput_kaloha(model, True, 10, i_units, Lambda)

    plt.title("Uso del canal")
    plt.ylabel('Uso del canal')
    plt.xlabel('Tasa de arribos')
    plt.grid(True)

    plt.plot(Lambda, s)
    plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)

    plt.plot(Lambda, s_)
    plt.fill_between(Lambda, ci_inf_, ci_sup_, color='r', alpha=.1)
    plt.legend(loc='upper right', labels=["Kaloha p=1", d.rsplit(".")[0]])
    #plt.fill_between(Lambda, ci_i, ci_s, color='r', alpha=.1)
    #plt.show()
    plt.savefig("img/kaloha/rnn/th-" + d.rsplit(".")[0] + ".png")
    plt.clf()

"""