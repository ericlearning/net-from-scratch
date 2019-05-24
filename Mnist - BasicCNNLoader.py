import numpy as np
import plotly
import plotly.graph_objs as go
from HypeNet.Networks.CNN_Simple import CNN_Simple
from HypeNet.Core.loadData import loadMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *
import os

np.random.seed(0)

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/MnistCNN/'

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadMnist(flatten = False)

network = CNN_Simple()
networkLoader(network, DIR)
acc = network.accuracy(X_val[:1000], Y_val[:, :1000])

print('--------Accuracy--------')
print(acc)
print('------------------------')