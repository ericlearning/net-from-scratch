import numpy as np
import plotly
import plotly.graph_objs as go
from HypeNet.Networks.CNN_Deep import CNN_Deep
from HypeNet.Core.loadData import loadMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *
import os

np.random.seed(0)

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/MnistCNN_Deep/'

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadMnist(flatten = False)

network = CNN_Deep()
networkLoader(network, DIR)
acc = network.accuracy(X_val, Y_val)

print('--------Accuracy--------')
print(acc)
print('------------------------')