import os
import numpy as np
from collections import OrderedDict
from HypeNet.Core.NetworkLayers import *
from HypeNet.Core.NetworkFunctions import cross_entropy_error

class FCNN_SoftmaxCE:
	def __init__(self, input_size, hidden_size_list, output_size, activation_list, weight_init_std = 'he', lambd = 0, use_dropout = False, keep_probs = [0.7, 0.7, 0.7, 0.7], use_batchnorm = False, batchnorm_params = {'momentum' : 0.9, 'epsilon' : 10e-7}, batchnorm_prev = True):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size_list = hidden_size_list
		self.hidden_layer_num = len(hidden_size_list)
		self.weight_init_std = weight_init_std
		self.lambd = lambd
		self.use_dropout = use_dropout
		self.use_batchnorm = use_batchnorm
		self.keep_probs = keep_probs
		self.activation_list = activation_list
		self.batchnorm_params = batchnorm_params	#batchnorm_params is only expected to have stuff from (momentum, epsilon)
		self.batchnorm_prev = batchnorm_prev

		self.params = {}
		self.unlearnable_params = {}
		self.paramInit()

		self.layers = OrderedDict()
		self.layerInit()

	def paramInit(self):
		all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
		for i in range(1, len(all_size_list)):
			if(self.weight_init_std == 'xavier'):
				self.params['W' + str(i)] = np.random.randn(all_size_list[i], all_size_list[i-1]) * np.sqrt(1 / all_size_list[i-1])
			elif(self.weight_init_std == 'he'):
				self.params['W' + str(i)] = np.random.randn(all_size_list[i], all_size_list[i-1]) * np.sqrt(2 / all_size_list[i-1])
			else:
				self.params['W' + str(i)] = np.random.randn(all_size_list[i], all_size_list[i-1]) * self.weight_init_std

			self.params['b' + str(i)] = np.zeros((all_size_list[i], 1))

		for i in range(1, self.hidden_layer_num + 1):		#No batchnorm in the output layer, because there's no activation in the last layer of this network, only softmax (Affine1 -> batchNorm1 -> Activation1 -> Dropout1 -> Affine2 -> ... AffineN -> SoftmaxCE)
			if(self.use_batchnorm == True):
				self.params['gamma' + str(i)] = np.ones((all_size_list[i], 1))
				self.params['beta' + str(i)] = np.zeros((all_size_list[i], 1))
		
		for i in range(1, self.hidden_layer_num + 1):
			if(self.use_batchnorm == True):
				self.unlearnable_params['running_mean' + str(i)] = None
				self.unlearnable_params['running_var' + str(i)] = None

	def paramInit_pretrained(self, pretrained_params, unlearnable_pretrained_params):
		all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
		for i in range(1, len(all_size_list)):
			self.params['W' + str(i)] = pretrained_params['W' + str(i)]
			self.params['b' + str(i)] = pretrained_params['b' + str(i)]

		for i in range(1, self.hidden_layer_num + 1):		#No batchnorm in the output layer, because there's no activation in the last layer of this network, only softmax (Affine1 -> batchNorm1 -> Activation1 -> Dropout1 -> Affine2 -> ... AffineN -> SoftmaxCE)
			if(self.use_batchnorm == True):
				self.params['gamma' + str(i)] = pretrained_params['gamma' + str(i)]
				self.params['beta' + str(i)] = pretrained_params['beta' + str(i)]
				self.unlearnable_params['running_mean' + str(i)] = unlearnable_pretrained_params['running_mean' + str(i)]
				self.unlearnable_params['running_var' + str(i)] = unlearnable_pretrained_params['running_var' + str(i)]

	def layerInit(self):
		for i in range(1, self.hidden_layer_num + 1):
			self.layers['Affine' + str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])

			if(self.batchnorm_prev == True):
				if(self.use_batchnorm == True):
					self.layers['BatchNorm' + str(i)] = BatchNormalization(self.params['gamma' + str(i)], self.params['beta' + str(i)], **self.batchnorm_params, running_mean = self.unlearnable_params['running_mean' + str(i)], running_var = self.unlearnable_params['running_var' + str(i)])

			if(self.activation_list[i-1].lower() == 'relu'):
				self.layers['Relu' + str(i)] = Relu()
			elif(self.activation_list[i-1].lower() == 'leakyrelu'):
				self.layers['LeakyRelu' + str(i)] = LeakyRelu()
			elif(self.activation_list[i-1].lower() == 'tanh'):
				self.layers['Tanh' + str(i)] = Tanh()
			elif(self.activation_list[i-1].lower() == 'sigmoid'):
				self.layers['Sigmoid' + str(i)] = Sigmoid()
			elif(self.activation_list[i-1].lower() == 'elu'):
				self.layers['Elu' + str(i)] = Elu()
			elif(self.activation_list[i-1].lower() == 'selu'):
				self.layers['Selu' + str(i)] = Selu()
			elif(self.activation_list[i-1].lower() == 'linear'):
				self.layers['Selu' + str(i)] = Linear()

			if(self.batchnorm_prev == False):
				if(self.use_batchnorm == True):
					self.layers['BatchNorm' + str(i)] = BatchNormalization(self.params['gamma' + str(i)], self.params['beta' + str(i)], **self.batchnorm_params, running_mean = self.unlearnable_params['running_mean' + str(i)], running_var = self.unlearnable_params['running_var' + str(i)])

			if(self.use_dropout == True):
				self.layers['Dropout' + str(i)] = Dropout(self.keep_probs[i-1])

		idx = self.hidden_layer_num + 1
		self.layers['Affine' + str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
		self.lastLayer = SoftmaxWithCrossEntropy()

	def predict(self, x, train_flg = False):
		for key, layer in self.layers.items():
			if('Dropout' in key):
				x = layer.forward(x, train_flg)
			if('BatchNorm' in key):
				x = layer.forward(x, train_flg)
			else:
				x = layer.forward(x)
			
		return x

	def loss(self, x, y, train_flg = False):
		m = x.shape[1]
		a = self.predict(x, train_flg)

		if(self.lambd == 0):
			return self.lastLayer.forward(a, y)
		else:	
			weight_decay_amount = 0
			for i in range(1, self.hidden_layer_num + 2):
				weight_decay_amount += np.sum(np.square(self.params['W' + str(i)]))

			weight_decay_amount *= self.lambd / (2 * m)
			return self.lastLayer.forward(a, y) + weight_decay_amount

	def gradient(self, x, y):
		m = x.shape[1]
		loss = self.loss(x, y, train_flg = True)
		dout = 1
		dout = self.lastLayer.backward(dout)
		
		layers = list(self.layers.values())
		layers.reverse()

		for index, layer in enumerate(layers):
			if(index == len(layers)-1):
				layer.backward(dout, calculate_dx = False)
				continue
			dout = layer.backward(dout)

		grads = {}
		for i in range(1, self.hidden_layer_num + 2):
			grads['W' + str(i)] = self.layers['Affine' + str(i)].dW + (self.lambd / m) * self.layers['Affine' + str(i)].W
			grads['b' + str(i)] = self.layers['Affine' + str(i)].db

			if(self.use_batchnorm == True and i != self.hidden_layer_num + 1):
				grads['gamma' + str(i)] = self.layers['BatchNorm' + str(i)].dgamma
				grads['beta' + str(i)] = self.layers['BatchNorm' + str(i)].dbeta
				
		return grads, loss

	def set_unlearnable_params(self):
		for i in range(1, self.hidden_layer_num + 1):
			if(self.use_batchnorm == True):
				self.unlearnable_params['running_mean' + str(i)] = self.layers['BatchNorm' + str(i)].running_mean
				self.unlearnable_params['running_var' + str(i)] = self.layers['BatchNorm' + str(i)].running_var

	def accuracy(self, x, y):
		a = self.predict(x)
		a_label = np.argmax(a, axis = 0)
		y_label = np.argmax(y, axis = 0)
		accuracy = np.mean(a_label == y_label) * 100

		return accuracy

	def returnParams(self):
		return self.params.copy(), self.unlearnable_params.copy()