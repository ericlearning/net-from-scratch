import numpy as np
import plotly
import plotly.graph_objs as go
from HypeNet.Networks.CNN_Simple import CNN_Simple
from HypeNet.Core.loadData import loadFashionMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *
import os

np.random.seed(0)

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/FashionMnistCNN/'

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadFashionMnist(flatten = False)
num_epoch = 1
minibatch_size = 50
save_network = True
learning_rate = 0.0005
optimizer_type = 'adam'

print('network created')
network = CNN_Simple()
print('network setting finished')
trainer = Trainer(network, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type, {'lr' : learning_rate}, verbose = True, LossAccInterval = 50)
train_loss_list, val_loss_list, train_acc_list, val_acc_list, x_axis, lrs = trainer.train()

networkSaver(network, DIR)