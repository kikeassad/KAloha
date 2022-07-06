from kaloha import crear_dataset_pktSlot
from kaloha import crear_dataset_l
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


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
model = Model(input_size=2, output_size=1, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 10001
lr=0.01

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


data = crear_dataset_pktSlot(1000)
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
plt.plot(range(len(val_loss)), val_loss,range(len(val_loss)), train_loss)
plt.legend(loc='upper right', labels=["val_loss", "train_loss"])
#plt.savefig("img/kaloha/rnn.png")
plt.show()
#torch.save(model.state_dict(), "modelos/rnn.mod")

#Later to restore:
#model.load_state_dict(torch.load(filepath))
#model.eval()