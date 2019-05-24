import numpy as np
import plotly
import plotly.graph_objs as go
from HypeNet.Networks.FCNN_SoftmaxCE import FCNN_SoftmaxCE
from HypeNet.Core.loadData import loadMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *
import os

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/Mnist/'

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadMnist()
num_epoch = 10
minibatch_size = 256
save_network = True
learning_rate = 0.001
optimizer_type = 'adam'

network = FCNN_SoftmaxCE(784, [256, 256, 256, 256], 10, ['Relu', 'Relu', 'Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = True, keep_probs = [0.9, 0.9, 0.9, 0.9])
trainer = Trainer(network, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type, {'lr' : learning_rate, 'epsilon' : 1e-8}, verbose = True, LossAccInterval = 200, LossAccOnNum = 'whole')
train_loss_list, val_loss_list, train_acc_list, val_acc_list, x_axis, lrs = trainer.train()

if(save_network == True):
	networkSaver(network, DIR)

trainLoss = go.Scatter(x = x_axis, y = train_loss_list, mode = 'lines', name = 'training loss')
valLoss = go.Scatter(x = x_axis, y = val_loss_list, mode = 'lines', name = 'validation loss')
trainAcc = go.Scatter(x = x_axis, y = train_acc_list, mode = 'lines', name = 'training acc')
valAcc = go.Scatter(x = x_axis, y = val_acc_list, mode = 'lines', name = 'validation acc')

loss_data = [trainLoss, valLoss]
acc_data = [trainAcc, valAcc]

plotly.offline.plot({'data' : loss_data, 'layout' : go.Layout(title = 'Loss')}, filename = 'Mnist_Loss.html')
plotly.offline.plot({'data' : acc_data, 'layout' : go.Layout(title = 'Accuracy')}, filename = 'Mnist_Acc.html')