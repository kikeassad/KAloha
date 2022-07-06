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


# Define hyperparameters
n_epochs = 50001 #poner en 50001
lr=0.001

h_dim = [2**x for x in range(4, 9)]
t_word = [2**x for x in range(5,9)]


for t in t_word:
    
    data_val = [crear_dataset_pktSlot(1000, tam_s=t) for _ in range(5)]
    
    for h in h_dim:
        """ """
        # Instantiate the model with hyperparameters
        model = Model(input_size=2, output_size=1, hidden_dim=h, n_layers=1)
        # We'll also set the model to the device that we defined earlier (default is CPU)
        model = model.to(device)

        # Define Loss, Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        data = crear_dataset_pktSlot(500000, tam_s=t)
        print("data created")


        x, y = crear_entrada(data)
        x_val, y_val = [], []
        for _ in range(5):
            x_temp, y_temp = crear_entrada(data_val[_])
            x_val.append(x_temp)
            y_val.append(y_temp)
        print("dead")

        input_seq = torch.FloatTensor(x)
        target_seq = torch.FloatTensor(y)
        
        val_input = []
        val_target = []

        for _ in range(5):
            val_input.append(torch.FloatTensor(x_val[_]))
            val_target.append(torch.FloatTensor(y_val[_]))

        train_loss = []
        val_loss = []

        i = []
        s = []

        perdida = 1000
        """ """
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
            
            if epoch%5000 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs))
                train_loss += [loss.item()]
                model.eval()
                temp_l = []
                for _ in range(5):
                    val_input[_] = val_input[_].to(device)
                    output, hidden = model(val_input[_])
                    output = output.to(device)
                    val_target[_] = val_target[_].to(device)
                    loss = criterion(output[:, -1].squeeze(), val_target[_])  
                    temp_l += [loss.item()] #checar que hacer aqui y como imprime el numpy.mean
                val_loss += [np.mean(temp_l)]
                inf, sup = st.t.interval(.95, len(temp_l) - 1, loc=np.mean(temp_l), scale=st.sem(temp_l))
                i.append(inf)
                s.append(sup)

                if loss.item() < perdida:
                    torch.save(model.state_dict(), "modelos/"+str(t)+"-"+str(h)+".mod")
                    perdida = loss.item()
                    
        plt.plot(range(len(val_loss)), val_loss,range(len(val_loss)), train_loss)
        plt.fill_between(range(len(val_loss)), i, s, color='r', alpha=.1)
        plt.legend(loc='upper right', labels=["val_loss", "train_loss"])
        plt.savefig("img/kaloha/rnn/"+str(t)+"_"+str(h)+".png")
        plt.clf()
