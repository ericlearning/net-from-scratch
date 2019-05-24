import numpy as np
import matplotlib.pyplot as plt
from HypeNet.Networks.FCNN_SoftmaxCE import FCNN_SoftmaxCE
import sklearn.datasets
from sklearn.model_selection import train_test_split
from HypeNet.Core.utils import *
from HypeNet.Core.Trainer import Trainer
import random

#number of points in the dataset
point_num = 100

#generate moon data, and split it
X, Y = sklearn.datasets.make_moons(n_samples = point_num, noise = 0.1)
X = X.T
Y = categorical(Y.reshape(1, point_num), 2)
X_train, X_val, Y_train, Y_val = train_test_split(X.T, Y.T, test_size = 0.2)

#correct the data shape (data processing finished)
X_train = X_train.T
X_val = X_val.T
Y_train = Y_train.T
Y_val = Y_val.T

#network settings
num_epoch = 100
minibatch_size = 64

#create network & trainer
network = FCNN_SoftmaxCE(2, [100, 100], 2, ['Relu', 'Relu'], weight_init_std = 'he')
trainer = Trainer(network, X_train, Y_train, X_val, Y_val, num_epoch, minibatch_size, 'adam', {'lr' : 0.01}, verbose = True, LossAccInterval = 10)

#start training network
train_loss_list, val_loss_list, train_acc_list, val_acc_list, x, lrs = trainer.train()
#network training finished

#feed the trained final network, whole moon data in order to visualize the result
draw_border(network, X_train, Y_train, X_val, Y_val, 0.01, c_map = 'viridis', c_edge = 'black')