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
save_network = False
learning_rate = 0.01
optimizer_type = 'adam'

network_prev = FCNN_SoftmaxCE(784, [1, 1], 10, ['Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = True, keep_probs = [0.9, 0.9], batchnorm_prev = True)
trainer_prev = Trainer(network_prev, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type, {'lr' : learning_rate}, verbose = True, LossAccInterval = 5)
train_loss_list_prev, val_loss_list_prev, train_acc_list_prev, val_acc_list_prev, x_axis, lrs = trainer_prev.train()

network_after = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = True, keep_probs = [0.9, 0.9], batchnorm_prev = False)
trainer_after = Trainer(network_after, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type, {'lr' : learning_rate}, verbose = True, LossAccInterval = 5)
train_loss_list_after, val_loss_list_after, train_acc_list_after, val_acc_list_after, x_axis, lrs = trainer_after.train()

network_none = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, use_batchnorm = False, keep_probs = [0.9, 0.9])
trainer_none = Trainer(network_none, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, optimizer_type, {'lr' : learning_rate}, verbose = True, LossAccInterval = 5)
train_loss_list_none, val_loss_list_none, train_acc_list_none, val_acc_list_none, x_axis, lrs = trainer_none.train()

trainLoss_prev = go.Scatter(x = x_axis, y = train_loss_list_prev, mode = 'lines', name = 'training loss prev', line = dict(color = ('rgb(232, 85, 85)')))
valLoss_prev = go.Scatter(x = x_axis, y = val_loss_list_prev, mode = 'lines', name = 'validation loss prev', line = dict(color = ('rgb(232, 85, 85)'), dash = 'dash'))
trainAcc_prev = go.Scatter(x = x_axis, y = train_acc_list_prev, mode = 'lines', name = 'training acc prev', line = dict(color = ('rgb(232, 85, 85)')))
valAcc_prev = go.Scatter(x = x_axis, y = val_acc_list_prev, mode = 'lines', name = 'validation acc prev', line = dict(color = ('rgb(232, 85, 85)'), dash = 'dash'))

trainLoss_after = go.Scatter(x = x_axis, y = train_loss_list_after, mode = 'lines', name = 'training loss after', line = dict(color = ('rgb(66, 134, 244)')))
valLoss_after = go.Scatter(x = x_axis, y = val_loss_list_after, mode = 'lines', name = 'validation loss after', line = dict(color = ('rgb(66, 134, 244)'), dash = 'dash'))
trainAcc_after = go.Scatter(x = x_axis, y = train_acc_list_after, mode = 'lines', name = 'training acc after', line = dict(color = ('rgb(66, 134, 244)')))
valAcc_after = go.Scatter(x = x_axis, y = val_acc_list_after, mode = 'lines', name = 'validation acc after', line = dict(color = ('rgb(66, 134, 244)'), dash = 'dash'))

trainLoss_none = go.Scatter(x = x_axis, y = train_loss_list_none, mode = 'lines', name = 'training loss none', line = dict(color = ('rgb(255, 189, 84)')))
valLoss_none = go.Scatter(x = x_axis, y = val_loss_list_none, mode = 'lines', name = 'validation loss none', line = dict(color = ('rgb(255, 189, 84)'), dash = 'dash'))
trainAcc_none = go.Scatter(x = x_axis, y = train_acc_list_none, mode = 'lines', name = 'training acc none', line = dict(color = ('rgb(255, 189, 84)')))
valAcc_none = go.Scatter(x = x_axis, y = val_acc_list_none, mode = 'lines', name = 'validation acc none', line = dict(color = ('rgb(255, 189, 84)'), dash = 'dash'))

loss_data = [trainLoss_prev, valLoss_prev, trainLoss_after, valLoss_after, trainLoss_none, valLoss_none]
acc_data = [trainAcc_prev, valAcc_prev, trainAcc_after, valAcc_after, trainAcc_none, valAcc_none]

plotly.offline.plot({'data' : loss_data, 'layout' : go.Layout(title = 'Loss')}, filename = 'FashionMnist_BNOrderTest_Loss.html')
plotly.offline.plot({'data' : acc_data, 'layout' : go.Layout(title = 'Accuracy')}, filename = 'FashionMnist_BNOrderTest_Acc.html')