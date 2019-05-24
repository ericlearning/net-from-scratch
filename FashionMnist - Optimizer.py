import numpy as np
import plotly
import plotly.graph_objs as go
from HypeNet.Networks.FCNN_SoftmaxCE import FCNN_SoftmaxCE
from HypeNet.Core.loadData import loadFashionMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *
import os

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/FashionMnist/'

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadFashionMnist()
num_epoch = 10
minibatch_size = 256
learning_rate = 0.05
optimizer_type1 = 'sgd'
optimizer_type2 = 'momentum'
optimizer_type3 = 'nesterov'

network1 = FCNN_SoftmaxCE(784, [256, 256, 256, 256, 256], 10, ['Relu', 'Relu', 'Relu', 'Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = True, keep_probs = [0.9, 0.9, 0.9, 0.9, 0.9])
trainer1 = Trainer(network1, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type1, {'lr' : learning_rate}, verbose = True, LossAccInterval = 20)
train_loss_list1, val_loss_list1, train_acc_list1, val_acc_list1, x_axis, lrs = trainer1.train()

network2 = FCNN_SoftmaxCE(784, [256, 256, 256, 256, 256], 10, ['Relu', 'Relu', 'Relu', 'Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = True, keep_probs = [0.9, 0.9, 0.9, 0.9, 0.9])
trainer2 = Trainer(network2, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type2, {'lr' : learning_rate}, verbose = True, LossAccInterval = 20)
train_loss_list2, val_loss_list2, train_acc_list2, val_acc_list2, x_axis, lrs = trainer2.train()

network3 = FCNN_SoftmaxCE(784, [256, 256, 256, 256, 256], 10, ['Relu', 'Relu', 'Relu', 'Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = True, keep_probs = [0.9, 0.9, 0.9, 0.9, 0.9])
trainer3 = Trainer(network3, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type3, {'lr' : learning_rate}, verbose = True, LossAccInterval = 20)
train_loss_list3, val_loss_list3, train_acc_list3, val_acc_list3, x_axis, lrs = trainer3.train()

trainLoss1 = go.Scatter(x = x_axis, y = train_loss_list1, mode = 'lines', name = 'training loss sgd', line = dict(color = ('rgb(232, 85, 85)')))
valLoss1 = go.Scatter(x = x_axis, y = val_loss_list1, mode = 'lines', name = 'validation loss sgd', line = dict(color = ('rgb(232, 85, 85)'), dash = 'dash'))
trainAcc1 = go.Scatter(x = x_axis, y = train_acc_list1, mode = 'lines', name = 'training acc sgd', line = dict(color = ('rgb(232, 85, 85)')))
valAcc1 = go.Scatter(x = x_axis, y = val_acc_list1, mode = 'lines', name = 'validation acc sgd', line = dict(color = ('rgb(232, 85, 85)'), dash = 'dash'))

trainLoss2 = go.Scatter(x = x_axis, y = train_loss_list2, mode = 'lines', name = 'training loss momentum', line = dict(color = ('rgb(66, 134, 244)')))
valLoss2 = go.Scatter(x = x_axis, y = val_loss_list2, mode = 'lines', name = 'validation loss momentum', line = dict(color = ('rgb(66, 134, 244)'), dash = 'dash'))
trainAcc2 = go.Scatter(x = x_axis, y = train_acc_list2, mode = 'lines', name = 'training acc momentum', line = dict(color = ('rgb(66, 134, 244)')))
valAcc2 = go.Scatter(x = x_axis, y = val_acc_list2, mode = 'lines', name = 'validation acc momentum', line = dict(color = ('rgb(66, 134, 244)'), dash = 'dash'))

trainLoss3 = go.Scatter(x = x_axis, y = train_loss_list3, mode = 'lines', name = 'training loss nesterov', line = dict(color = ('rgb(200, 200, 200)')))
valLoss3 = go.Scatter(x = x_axis, y = val_loss_list3, mode = 'lines', name = 'validation loss nesterov', line = dict(color = ('rgb(200, 200, 200)'), dash = 'dash'))
trainAcc3 = go.Scatter(x = x_axis, y = train_acc_list3, mode = 'lines', name = 'training acc nesterov', line = dict(color = ('rgb(200, 200, 200)')))
valAcc3 = go.Scatter(x = x_axis, y = val_acc_list3, mode = 'lines', name = 'validation acc nesterov', line = dict(color = ('rgb(200, 200, 200)'), dash = 'dash'))

loss_data = [trainLoss1, valLoss1, trainLoss2, valLoss2, trainLoss3, valLoss3]
acc_data = [trainAcc1, valAcc1, trainAcc2, valAcc2, trainAcc3, valAcc3]

plotly.offline.plot({'data' : loss_data, 'layout' : go.Layout(title = 'Loss')}, filename = 'FashionMnist_OptLoss.html')
plotly.offline.plot({'data' : acc_data, 'layout' : go.Layout(title = 'Accuracy')}, filename = 'FashionMnist_OptAcc.html')