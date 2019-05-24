import numpy as np
import plotly
from HypeNet.Networks.FCNN_SoftmaxCE import FCNN_SoftmaxCE
import plotly.graph_objs as go
from HypeNet.Core.loadData import loadFashionMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadFashionMnist()
X_train_aug, Y_train_aug = data_augmentation(X_train, Y_train, rotate = 5.0, image_size = (28, 28))
X_train_extended, Y_train_extended = data_augmentation(X_train, Y_train, rotate = 5.0, image_size = (28, 28))
num_epoch = 5
minibatch_size = 256

network_augment = FCNN_SoftmaxCE(784, [256, 256, 256, 256], 10, ['Relu', 'Relu', 'Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, keep_probs = [0.9, 0.9, 0.9, 0.9])
network_increase = FCNN_SoftmaxCE(784, [256, 256, 256, 256], 10, ['Relu', 'Relu', 'Relu', 'Relu'], weight_init_std = 'he', use_dropout = True, keep_probs = [0.9, 0.9, 0.9, 0.9])

trainer_augment = Trainer(network_augment, X_train_aug, Y_train_aug, X_val, Y_val, num_epoch, minibatch_size, 'adam', {'lr' : 0.001}, verbose = True, LossAccInterval = 100)
trainer_increase = Trainer(network_increase, X_train_extended, Y_train_extended, X_val, Y_val, num_epoch, minibatch_size, 'adam', {'lr' : 0.001}, verbose = True, LossAccInterval = 100)

train_loss_list_augment, val_loss_list_augment, train_acc_list_augment, val_acc_list_augment, x, lrs = trainer_augment.train()
train_loss_list_increase, val_loss_list_increase, train_acc_list_increase, val_acc_list_increase, x, lrs = trainer_increase.train()

trainLoss_aug = go.Scatter(x = x, y = train_loss_list_augment, mode = 'lines', name = 'augment', line = dict(color = ('rgb(232, 85, 85)')))
valLoss_aug = go.Scatter(x = x, y = val_loss_list_augment, mode = 'lines', name = 'augment', line = dict(color = ('rgb(232, 85, 85)'), dash = 'dash'))
trainAcc_aug = go.Scatter(x = x, y = train_acc_list_augment, mode = 'lines', name = 'augment', line = dict(color = ('rgb(232, 85, 85)')))
valAcc_aug = go.Scatter(x = x, y = val_acc_list_augment, mode = 'lines', name = 'augment', line = dict(color = ('rgb(232, 85, 85)'), dash = 'dash'))

trainLoss_inc = go.Scatter(x = x, y = train_loss_list_increase, mode = 'lines', name = 'increase', line = dict(color = ('rgb(66, 134, 244)')))
valLoss_inc = go.Scatter(x = x, y = val_loss_list_increase, mode = 'lines', name = 'increase', line = dict(color = ('rgb(66, 134, 244)'), dash = 'dash'))
trainAcc_inc = go.Scatter(x = x, y = train_acc_list_increase, mode = 'lines', name = 'increase', line = dict(color = ('rgb(66, 134, 244)')))
valAcc_inc = go.Scatter(x = x, y = val_acc_list_increase, mode = 'lines', name = 'increase', line = dict(color = ('rgb(66, 134, 244)'), dash = 'dash'))


loss_data = [trainLoss_aug, valLoss_aug, trainLoss_inc, valLoss_inc]
acc_data = [trainAcc_aug, valAcc_aug, trainAcc_inc, valAcc_inc]

plotly.offline.plot({'data' : loss_data, 'layout' : go.Layout(title = 'Loss')}, filename = 'FashionMnist_Augment_Loss.html')
plotly.offline.plot({'data' : acc_data, 'layout' : go.Layout(title = 'Accuracy')}, filename = 'FashionMnist_Augment_Acc.html')