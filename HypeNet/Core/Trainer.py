import numpy as np
from HypeNet.Core.optimizer import *
from HypeNet.Core.CreateBatch import createBatch
from HypeNet.Core.LrSchedule import *
import matplotlib.pyplot as plt

class Trainer:
	def __init__(self, network, x_train, y_train, x_val, y_val, epochs = 20, minibatch_size = -1, optimizer_type = 'SGD', optimizer_params = {'lr' : 0.01}, lr_scheduler_type = 'Linear', lr_scheduler_params = None, verbose = True, LossAccInterval = 1, LossAccOnNum = 1000, print_iteration = False):
		self.network = network
		self.X_train = x_train
		self.Y_train = y_train
		self.X_val = x_val
		self.Y_val = y_val

		self.epochs = epochs
		self.minibatch_size = minibatch_size
		if(self.minibatch_size == -1):
			self.minibatch_size = self.X_train.shape[1]

		optimizer_class_dict = {'sgd' : SGD, 'momentum' : Momentum, 'nesterov' : Nesterov, 'rmsprop' : RmsProp, 'adam' : Adam, 'amsgrad' : AmsGrad}
		lr_scheduler_dict = {'linear' : LinearDecay, 'exp_decay' : ExpDecay, 'div_decay' : DivDecay, 'triangular' : Triangular, 'triangular2' : Triangular_2, 'triangularexp' : Triangular_exp}

		self.optimizer = optimizer_class_dict[optimizer_type.lower()](**optimizer_params)
		if(lr_scheduler_params == None):
			self.lr_scheduler = lr_scheduler_dict[lr_scheduler_type.lower()]()
		else:
			self.lr_scheduler = lr_scheduler_dict[lr_scheduler_type.lower()](**lr_scheduler_params)
		self.lr_scheduler.set_lr(optimizer_params['lr'])

		self.train_loss_list = []
		self.val_loss_list = []
		self.train_acc_list = []
		self.val_acc_list = []

		self.minibatches = createBatch(self.X_train, self.Y_train, self.minibatch_size)
		self.minibatches_num = len(self.minibatches)
		self.verbose = verbose

		self.LossAccInterval = LossAccInterval
		self.LossAccOnNum = LossAccOnNum
		self.print_iteration = print_iteration

		self.dataDim = self.X_train.ndim

		if(self.LossAccOnNum == 'whole'):
			self.X_train_for_LossAcc = self.X_train
			self.Y_train_for_LossAcc = self.Y_train
			self.X_val_for_LossAcc = self.X_val
			self.Y_val_for_LossAcc = self.Y_val

		else:
			if(self.dataDim == 4):
				self.X_train_for_LossAcc = self.X_train[:self.LossAccOnNum]
				self.Y_train_for_LossAcc = self.Y_train[:, :self.LossAccOnNum]
				self.X_val_for_LossAcc = self.X_val[:self.LossAccOnNum]
				self.Y_val_for_LossAcc = self.Y_val[:, :self.LossAccOnNum]

			elif(self.dataDim == 2):
				self.X_train_for_LossAcc = self.X_train[:, :self.LossAccOnNum]
				self.Y_train_for_LossAcc = self.Y_train[:, :self.LossAccOnNum]
				self.X_val_for_LossAcc = self.X_val[:, :self.LossAccOnNum]
				self.Y_val_for_LossAcc = self.Y_val[:, :self.LossAccOnNum]


	def train_step(self, cur_minibatch, calculate_LossAcc = True):
		(X_train_minibatch, Y_train_minibatch) = cur_minibatch
		
		grad, loss_minibatch = self.network.gradient(X_train_minibatch, Y_train_minibatch)
		loss, loss_val, acc, acc_val = None, None, None, None

		if(calculate_LossAcc == True):
			loss = self.network.loss(self.X_train_for_LossAcc, self.Y_train_for_LossAcc)
			self.train_loss_list.append(loss)
			loss_val = self.network.loss(self.X_val_for_LossAcc, self.Y_val_for_LossAcc)
			self.val_loss_list.append(loss_val)
			acc = self.network.accuracy(self.X_train_for_LossAcc, self.Y_train_for_LossAcc)
			self.train_acc_list.append(acc)
			acc_val = self.network.accuracy(self.X_val_for_LossAcc, self.Y_val_for_LossAcc)
			self.val_acc_list.append(acc_val)

		params = self.network.params
		self.optimizer.update(params, grad)
		
		return loss, loss_val, acc, acc_val

	def train(self):
		cnt = 1
		x_axis = []
		cur_lrs = []
		for i in range(self.epochs):
			for k, minibatch in enumerate(self.minibatches):

				if(self.print_iteration == True):
					print('{0} Iteration Training...'.format(cnt))

				self.decay_lr(cnt-1)
				if(k % self.LossAccInterval == 0):
					loss, loss_val, acc, acc_val = self.train_step(minibatch, calculate_LossAcc = True)
					x_axis.append(cnt)
					cur_lrs.append(self.optimizer.lr)
				else:
					loss, loss_val, acc, acc_val = self.train_step(minibatch, calculate_LossAcc = False)
				if(self.verbose == True):
					if(k % self.LossAccInterval == 0):
						print('(Epoch : {0} / {1}) / (Iteration : {2} / {3}) / Loss : {4:0.5f} / Loss_val : {5:0.5f} / Acc : {6:0.5f} / Acc_val : {7:0.5f} / current_lr : {8:0.7f}'.format(i+1, self.epochs, k+1, self.minibatches_num, loss, loss_val, acc, acc_val, self.optimizer.lr))
				cnt+=1

		self.network.set_unlearnable_params()
		return self.train_loss_list, self.val_loss_list, self.train_acc_list, self.val_acc_list, x_axis, cur_lrs

	def decay_lr(self, iteration):
		updated_lr = self.lr_scheduler.update_lr(iteration)
		self.optimizer.lr = updated_lr
		