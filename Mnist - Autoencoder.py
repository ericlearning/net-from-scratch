import numpy as np
import matplotlib.pyplot as plt
from HypeNet.Networks.FCNN_MSE import FCNN_MSE
from HypeNet.Core.loadData import loadMnist
from HypeNet.Core.Trainer import Trainer
from HypeNet.Core.utils import *

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadMnist()
num_epoch = 20
minibatch_size = 256

network = FCNN_MSE(784, [1000, 500, 100, 500, 1000], 784, ['Relu', 'Relu', 'Relu', 'Relu', 'Relu', 'Sigmoid'], weight_init_std = 'he', use_dropout = False, keep_probs = [0.9, 0.9, 0.9, 0.9, 0.9], use_batchnorm = False)
trainer = Trainer(network, X_train, X_train, X_val, X_val, num_epoch, minibatch_size, 'adam', {'lr' : 0.0004}, verbose = True, LossAccInterval = 10000, lr_scheduler_type = 'exp_decay', lr_scheduler_params = {'k' : 0.00001})
train_loss_list, val_loss_list, train_acc_list, val_acc_list, x, lrs = trainer.train()

visualize_example = X_val.T[:25].T 					#(784, 25)
reconstructed_example = network.predict(visualize_example)	#(784, 25)

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/MnistAutoencoder/'
networkSaver(network, DIR)

x_num, y_num = 5, 5

fig1 = plt.figure('Given Image', figsize = (8, 8))
for i in range(1, x_num * y_num + 1):
	cur_img = visualize_example.T[i-1].reshape(28, 28)
	fig1.add_subplot(x_num, y_num, i)
	plt.imshow(cur_img)

fig2 = plt.figure('Reconstructed Image', figsize = (8, 8))
for i in range(1, x_num * y_num + 1):
	cur_img = reconstructed_example.T[i-1].reshape(28, 28)
	fig2.add_subplot(x_num, y_num, i)
	plt.imshow(cur_img)

plt.show()