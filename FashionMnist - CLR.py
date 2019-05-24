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
stepsize = math.ceil(X_train.shape[1] / minibatch_size) * 2
cycle_num = iteration_num / (stepsize * 2)

network1 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network2 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network3 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network4 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network5 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network6 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network7 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network8 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network9 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')
network10 = FCNN_SoftmaxCE(784, [256, 256], 10, ['Relu', 'Relu'], weight_init_std = 'he')

trainer1 = Trainer(network1, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : -1}, lr_scheduler_type = 'triangular', lr_scheduler_params = {'stepsize' : stepsize, 'base_lr' : 0.0001, 'max_lr' : 0.35}, verbose = True, LossAccInterval = 10)
trainer2 = Trainer(network2, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : -1}, lr_scheduler_type = 'triangular2', lr_scheduler_params = {'stepsize' : stepsize, 'base_lr' : 0.0001, 'max_lr' : 0.35}, verbose = True, LossAccInterval = 10)
trainer3 = Trainer(network3, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : -1}, lr_scheduler_type = 'triangularExp', lr_scheduler_params = {'stepsize' : stepsize, 'base_lr' : 0.0001, 'max_lr' : 0.35}, verbose = True, LossAccInterval = 10)
trainer4 = Trainer(network4, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.0001}, verbose = True, LossAccInterval = 10)
trainer5 = Trainer(network5, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.0581}, verbose = True, LossAccInterval = 10)
trainer6 = Trainer(network6, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.1161}, verbose = True, LossAccInterval = 10)
trainer7 = Trainer(network7, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.1741}, verbose = True, LossAccInterval = 10)
trainer8 = Trainer(network8, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.2321}, verbose = True, LossAccInterval = 10)
trainer9 = Trainer(network9, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.2901}, verbose = True, LossAccInterval = 10)
trainer10 = Trainer(network10, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'sgd', {'lr' : 0.35}, verbose = True, LossAccInterval = 10)

train_loss_list_1, val_loss_list_1, train_acc_list_1, val_acc_list_1, x, lrs_1 = trainer1.train()
train_loss_list_2, val_loss_list_2, train_acc_list_2, val_acc_list_2, x, lrs_2 = trainer2.train()
train_loss_list_3, val_loss_list_3, train_acc_list_3, val_acc_list_3, x, lrs_3 = trainer3.train()
train_loss_list_4, val_loss_list_4, train_acc_list_4, val_acc_list_4, x, lrs_4 = trainer4.train()
train_loss_list_5, val_loss_list_5, train_acc_list_5, val_acc_list_5, x, lrs_5 = trainer5.train()
train_loss_list_6, val_loss_list_6, train_acc_list_6, val_acc_list_6, x, lrs_6 = trainer6.train()
train_loss_list_7, val_loss_list_7, train_acc_list_7, val_acc_list_7, x, lrs_7 = trainer7.train()
train_loss_list_8, val_loss_list_8, train_acc_list_8, val_acc_list_8, x, lrs_8 = trainer8.train()
train_loss_list_9, val_loss_list_9, train_acc_list_9, val_acc_list_9, x, lrs_9 = trainer9.train()
train_loss_list_10, val_loss_list_10, train_acc_list_10, val_acc_list_10, x, lrs_10 = trainer10.train()

trainLoss_1 = go.Scatter(x = x, y = train_loss_list_1, mode = 'lines', name = 'triangular')
trainLoss_2 = go.Scatter(x = x, y = train_loss_list_2, mode = 'lines', name = 'triangular2')
trainLoss_3 = go.Scatter(x = x, y = train_loss_list_3, mode = 'lines', name = 'triangularExp')
trainLoss_4 = go.Scatter(x = x, y = train_loss_list_4, mode = 'lines', name = '0.0001')
trainLoss_5 = go.Scatter(x = x, y = train_loss_list_5, mode = 'lines', name = '0.0581')
trainLoss_6 = go.Scatter(x = x, y = train_loss_list_6, mode = 'lines', name = '0.1161')
trainLoss_7 = go.Scatter(x = x, y = train_loss_list_7, mode = 'lines', name = '0.1741')
trainLoss_8 = go.Scatter(x = x, y = train_loss_list_8, mode = 'lines', name = '0.2321')
trainLoss_9 = go.Scatter(x = x, y = train_loss_list_9, mode = 'lines', name = '0.2901')
trainLoss_10 = go.Scatter(x = x, y = train_loss_list_10, mode = 'lines', name = '0.35')

valLoss_1 = go.Scatter(x = x, y = val_loss_list_1, mode = 'lines', name = 'triangular')
valLoss_2 = go.Scatter(x = x, y = val_loss_list_2, mode = 'lines', name = 'triangular2')
valLoss_3 = go.Scatter(x = x, y = val_loss_list_3, mode = 'lines', name = 'triangularExp')
valLoss_4 = go.Scatter(x = x, y = val_loss_list_4, mode = 'lines', name = '0.0001')
valLoss_5 = go.Scatter(x = x, y = val_loss_list_5, mode = 'lines', name = '0.0581')
valLoss_6 = go.Scatter(x = x, y = val_loss_list_6, mode = 'lines', name = '0.1161')
valLoss_7 = go.Scatter(x = x, y = val_loss_list_7, mode = 'lines', name = '0.1741')
valLoss_8 = go.Scatter(x = x, y = val_loss_list_8, mode = 'lines', name = '0.2321')
valLoss_9 = go.Scatter(x = x, y = val_loss_list_9, mode = 'lines', name = '0.2901')
valLoss_10 = go.Scatter(x = x, y = val_loss_list_10, mode = 'lines', name = '0.35')

trainAcc_1 = go.Scatter(x = x, y = train_acc_list_1, mode = 'lines', name = 'triangular')
trainAcc_2 = go.Scatter(x = x, y = train_acc_list_2, mode = 'lines', name = 'triangular2')
trainAcc_3 = go.Scatter(x = x, y = train_acc_list_3, mode = 'lines', name = 'triangularExp')
trainAcc_4 = go.Scatter(x = x, y = train_acc_list_4, mode = 'lines', name = '0.0001')
trainAcc_5 = go.Scatter(x = x, y = train_acc_list_5, mode = 'lines', name = '0.0581')
trainAcc_6 = go.Scatter(x = x, y = train_acc_list_6, mode = 'lines', name = '0.1161')
trainAcc_7 = go.Scatter(x = x, y = train_acc_list_7, mode = 'lines', name = '0.1741')
trainAcc_8 = go.Scatter(x = x, y = train_acc_list_8, mode = 'lines', name = '0.2321')
trainAcc_9 = go.Scatter(x = x, y = train_acc_list_9, mode = 'lines', name = '0.2901')
trainAcc_10 = go.Scatter(x = x, y = train_acc_list_10, mode = 'lines', name = '0.35')

valAcc_1 = go.Scatter(x = x, y = val_acc_list_1, mode = 'lines', name = 'triangular')
valAcc_2 = go.Scatter(x = x, y = val_acc_list_2, mode = 'lines', name = 'triangular2')
valAcc_3 = go.Scatter(x = x, y = val_acc_list_3, mode = 'lines', name = 'triangularExp')
valAcc_4 = go.Scatter(x = x, y = val_acc_list_4, mode = 'lines', name = '0.0001')
valAcc_5 = go.Scatter(x = x, y = val_acc_list_5, mode = 'lines', name = '0.0581')
valAcc_6 = go.Scatter(x = x, y = val_acc_list_6, mode = 'lines', name = '0.1161')
valAcc_7 = go.Scatter(x = x, y = val_acc_list_7, mode = 'lines', name = '0.1741')
valAcc_8 = go.Scatter(x = x, y = val_acc_list_8, mode = 'lines', name = '0.2321')
valAcc_9 = go.Scatter(x = x, y = val_acc_list_9, mode = 'lines', name = '0.2901')
valAcc_10 = go.Scatter(x = x, y = val_acc_list_10, mode = 'lines', name = '0.35')

trainLoss_data = [trainLoss_1, trainLoss_2, trainLoss_3, trainLoss_4, trainLoss_5, trainLoss_6, trainLoss_7, trainLoss_8, trainLoss_9, trainLoss_10]
valLoss_data = [valLoss_1, valLoss_2, valLoss_3, valLoss_4, valLoss_5, valLoss_6, valLoss_7, valLoss_8, valLoss_9, valLoss_10]
trainAcc_data = [trainAcc_1, trainAcc_2, trainAcc_3, trainAcc_4, trainAcc_5, trainAcc_6, trainAcc_7, trainAcc_8, trainAcc_9, trainAcc_10]
valAcc_data = [valAcc_1, valAcc_2, valAcc_3, valAcc_4, valAcc_5, valAcc_6, valAcc_7, valAcc_8, valAcc_9, valAcc_10]

plotly.offline.plot({'data' : trainLoss_data, 'layout' : go.Layout(title = 'Training Loss')}, filename = 'FashionMnist_CLR_trainLoss.html')
plotly.offline.plot({'data' : valLoss_data, 'layout' : go.Layout(title = 'Validation Loss')}, filename = 'FashionMnist_CLR_valLoss.html')
plotly.offline.plot({'data' : trainAcc_data, 'layout' : go.Layout(title = 'Training Accuracy')}, filename = 'FashionMnist_CLR_trainAcc.html')
plotly.offline.plot({'data' : valAcc_data, 'layout' : go.Layout(title = 'Validation Accuracy')}, filename = 'FashionMnist_CLR_valAcc.html')
