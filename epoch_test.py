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
        #train_split=None,
        warm_start=True,
    )
    return regressor


def entrenar_modelos(ambiente, i_units, h_units, h_layers, src="modelos/"):
    
    col = [True, False]
    name = str(i_units) + "_" + str(h_layers) + "_" + str(h_units)
    for i in col:
        reg = crear_modelo(i_units, h_units, h_layers)
        l_data = 0
        hist = {}
        E_in = []
        E_out = []
        x_error = []
        print(l_data)
        data = ambiente.crear_dataset_l(50000, channel=i, tam_s=i_units)
        Y = data[:, -1]
        Y = Y.reshape(-1, 1)
        Y = Y.astype(np.float32)
        X = data[:, :-1]
        Y = Y.reshape(-1, 1)
        X = X.astype(np.float32)
        reg.fit(X, Y)

        history = reg.history
        print(history[:, 'valid_loss'])

        del reg
        del X
        del Y

entrenar_modelos(kaloha, 16, 13, 31)