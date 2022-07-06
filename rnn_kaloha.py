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


torch.manual_seed(13)
np.random.seed(17)

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


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Instantiate the model with hyperparameters
model = Model(input_size=2, output_size=1, hidden_dim=32, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 20001
lr=0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


data = crear_dataset_pktSlot(10000)
data_val = crear_dataset_pktSlot(1000)
print("data created")


x, y = crear_entrada(data)
x_val, y_val = crear_entrada(data_val)

input_seq = torch.FloatTensor(x)
target_seq = torch.FloatTensor(y)

val_input = torch.FloatTensor(x_val)
val_target = torch.FloatTensor(y_val)

train_loss = []
val_loss = []

perdida = 1000
"""
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    input_seq = input_seq.to(device)
    output, hidden = model(input_seq)
    output = output.to(device)
    target_seq = target_seq.to(device)
    loss = criterion(output[:, -1].squeeze(), target_seq)    
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%1000 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs))
        train_loss += [loss.item()]
        model.eval()
        val_input = val_input.to(device)
        output, hidden = model(val_input)
        output = output.to(device)
        val_target = val_target.to(device)
        loss = criterion(output[:, -1].squeeze(), val_target)  
        val_loss += [loss.item()]
        if loss.item() < perdida:
            torch.save(model.state_dict(), "modelos/rnn.mod")
            perdida = loss.item()

plt.plot(range(len(val_loss)), val_loss,range(len(val_loss)), train_loss)
plt.legend(loc='upper right', labels=["val_loss", "train_loss"])
plt.savefig("img/kaloha/rnn.png")
plt.show()
"""

#Later to restore:
#model.load_state_dict(torch.load(filepath))
#model.eval()

def predict(modelo, entrada, dev):
    modelo.eval()
    valor = torch.FloatTensor(entrada)
    valor = torch.unsqueeze(valor, 0)
    valor = valor.to(device)
    output, hidden = modelo(valor)
    output = output.to(device="cpu").detach().numpy()

    return np.squeeze(output)[-1]

    

""" 
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
"""

def prob(l):
    if l == 0:
        return 1
    return np.minimum(1.0, 1.0/l)


def throughput_kaloha(modelo, channel, n_sim, tam_s, Lambda, p=1, bps=1e6, t_pkt="exp", delay=0, n_nodos=61):
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
                    S.push(Event(t + slot, "intra_slot", -1), t + slot)
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
                    for x in nodos:
                        x.n_pkts += 1
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

            while t < 50 + t_lim:
                e = S.pop()
                t = e.tiempo
                n = e.nodo
                tipo = e.tipo

                if tipo == "intra_slot":
                    S.push(Event(t + slot, "intra_slot", -1), t + slot )
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
                        val = predict(modelo, seq, device)
                        p = prob(val)
                        nodos[n].set_prob(p)
                        
                        if nodos[n].start_tx(t, S):
                            C.ocupar()

                if tipo == "StartTx":
                    for x in nodos:
                        x.n_pkts += 1
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

            aux.append(m/(50))
        result.append(np.mean(aux))
        inf, sup = st.t.interval(.95, len(aux) - 1, loc=np.mean(aux), scale=st.sem(aux))
        ci_sup.append(sup)
        ci_inf.append(inf)

    return result, ci_inf, ci_sup

# Cargamos el modelo
model.load_state_dict(torch.load("modelos/rnn.mod"))

Lambda = np.arange(0.1, 6.0, .1)
i_units = 8
#

s, ci_inf, ci_sup = throughput(75, Lambda)# prob = 1

s_, ci_inf_, ci_sup_ = throughput_kaloha(model, True, 10, i_units, Lambda)

plt.title("Uso del canal")
plt.ylabel('Uso del canal')
plt.xlabel('Tasa de arribos')
plt.grid(True)

plt.plot(Lambda, s)
#plt.legend(loc='upper right', labels=[d.rsplit(".")[0], "Kaloha p=1"])
plt.fill_between(Lambda, ci_inf, ci_sup, color='b', alpha=.1)

plt.plot(Lambda, s_)
plt.fill_between(Lambda, ci_inf_, ci_sup_, color='r', alpha=.1)
#plt.fill_between(Lambda, ci_i, ci_s, color='r', alpha=.1)
plt.show()
#plt.savefig("img/kaloha/" + d.rsplit(".")[0] + ".png")