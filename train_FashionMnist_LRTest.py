import numpy as np
import plotly
import plotly.graph_objs as go
from HypeNet.Networks.FCNN_SoftmaxCE import FCNN_SoftmaxCE
from HypeNet.Core.loadData import loadFashionMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *
import math

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadFashionMnist()
num_epoch = 20
minibatch_size = 256
iteration_num = math.ceil(X_train.shape[1] / minibatch_size) * num_epoch
stepsize = iteration_num
cycle_num = iteration_num / (stepsize * 2)

network = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
trainer = Trainer(network, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : -1}, lr_scheduler_type = 'triangular', lr_scheduler_params = {'stepsize' : stepsize, 'base_lr' : 0.0001, 'max_lr' : 0.5}, verbose = True, LossAccInterval = 10)
train_loss_list, val_loss_list, train_acc_list, val_acc_list, x_axis, lrs = trainer.train()

trainLoss = go.Scatter(x = x_axis, y = train_loss_list, mode = 'lines', name = 'training loss')
valLoss = go.Scatter(x = x_axis, y = val_loss_list, mode = 'lines', name = 'validation loss')
trainAcc = go.Scatter(x = x_axis, y = train_acc_list, mode = 'lines', name = 'training acc')
valAcc = go.Scatter(x = x_axis, y = val_acc_list, mode = 'lines', name = 'validation acc')
lr_change = go.Scatter(x = x_axis, y = lrs, mode = 'lines', name = 'learning rate')

loss_data = [trainLoss, valLoss, lr_change]
acc_data = [trainAcc, valAcc, lr_change]

plotly.offline.plot({'data' : loss_data, 'layout' : go.Layout(title = 'Loss')}, filename = 'FashionMnist_LRTest_Loss.html')
plotly.offline.plot({'data' : acc_data, 'layout' : go.Layout(title = 'Accuracy')}, filename = 'FashionMnist_LRTest_Acc.html')