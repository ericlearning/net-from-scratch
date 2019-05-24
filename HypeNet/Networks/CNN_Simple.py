import os
import numpy as np
from collections import OrderedDict
from HypeNet.Core.NetworkLayers import *
from HypeNet.Core.NetworkFunctions import cross_entropy_error
from HypeNet.Core.utils import calculateConvOutputSize

'''
[Architecture]
	Input -> Conv(16,5,5) -> Relu -> Pool
		  -> Conv(32,5,5) -> Relu -> Pool
		  -> Affine(120) -> Relu -> Affine(10) -> Softmax
'''
class CNN_Simple:
	def __init__(self, input_dim = (1, 28, 28), conv_param1 = {'filter_num' : 16, 'filter_height' : 5, 'filter_width' : 5, 'pad' : 2, 'stride' : 1}
											  , conv_param2 = {'filter_num' : 32, 'filter_height' : 5, 'filter_width' : 5, 'pad' : 2, 'stride' : 1}
											  , hidden_size = 200, output_size = 10):

		print('[Open]')
		self.input_dim = input_dim
		self.conv_param1 = conv_param1
		self.conv_param2 = conv_param2
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.unlearnable_params = {}
		self.params = {}
		self.paramInit()

		self.layers = OrderedDict()
		self.layerInit()

	def calculate_final_conv_output_size(self):
		input_H, input_W = self.input_dim[1], self.input_dim[2]

		conv1_output_H = calculateConvOutputSize(input_H, self.conv_param1['pad'], self.conv_param1['filter_height'], self.conv_param1['stride'])
		pool1_output_H = conv1_output_H // 2
		conv2_output_H = calculateConvOutputSize(pool1_output_H, self.conv_param2['pad'], self.conv_param2['filter_height'], self.conv_param2['stride'])
		pool2_output_H = conv2_output_H // 2

		conv1_output_W = calculateConvOutputSize(input_W, self.conv_param1['pad'], self.conv_param1['filter_width'], self.conv_param1['stride'])
		pool1_output_W = conv1_output_W // 2
		conv2_output_W = calculateConvOutputSize(pool1_output_W, self.conv_param2['pad'], self.conv_param2['filter_width'], self.conv_param2['stride'])
		pool2_output_W = conv2_output_W // 2

		final_conv_output_H = pool2_output_H
		final_conv_output_W = pool2_output_W

		return final_conv_output_H, final_conv_output_W

	def paramInit(self):
		(final_conv_output_H, final_conv_output_W), final_conv_output_C = self.calculate_final_conv_output_size(), self.conv_param2['filter_num']
		
		prev_connect_num = [self.input_dim[0] * self.conv_param1['filter_height'] * self.conv_param1['filter_width'], 
							self.conv_param1['filter_num'] * self.conv_param2['filter_height'] * self.conv_param2['filter_width'], 
							final_conv_output_H * final_conv_output_W * final_conv_output_C, self.hidden_size]

		self.params['W1'] = np.random.randn(self.conv_param1['filter_num'], self.input_dim[0], self.conv_param1['filter_height'], self.conv_param1['filter_width']) * np.sqrt(2.0 / prev_connect_num[0])
		self.params['b1'] = np.zeros((self.conv_param1['filter_num'], 1))
		self.params['W2'] = np.random.randn(self.conv_param2['filter_num'], self.conv_param1['filter_num'], self.conv_param2['filter_height'], self.conv_param2['filter_width']) * np.sqrt(2.0 / prev_connect_num[1])
		self.params['b2'] = np.zeros((self.conv_param2['filter_num'], 1))
		self.params['W3'] = np.random.randn(prev_connect_num[3], prev_connect_num[2]) * np.sqrt(2.0 / prev_connect_num[2])
		self.params['b3'] = np.zeros((prev_connect_num[3], 1))
		self.params['W4'] = np.random.randn(self.output_size, prev_connect_num[3]) * np.sqrt(2.0 / prev_connect_num[3])
		self.params['b4'] = np.zeros((self.output_size, 1))

	def paramInit_pretrained(self, pretrained_params, unlearnable_pretrained_params):
		for i in range(4):
			self.params['W' + str(i+1)] = pretrained_params['W' + str(i+1)]
			self.params['b' + str(i+1)] = pretrained_params['b' + str(i+1)]

	def layerInit(self):
		self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], self.conv_param1['stride'], self.conv_param1['pad'])
		self.layers['Relu1'] = Relu()
		self.layers['Pool1'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)
		self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], self.conv_param2['stride'], self.conv_param2['pad'])
		self.layers['Relu2'] = Relu()
		self.layers['Pool2'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)
		self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
		self.layers['Relu3'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
		self.lastLayer = SoftmaxWithCrossEntropy()

	def predict(self, x, train_flg = False):
		cnt = 0
		for key, layer in self.layers.items():
			cnt+=1
			if('Dropout' in key):
				x = layer.forward(x, train_flg)
			else:
				x = layer.forward(x)
			
		return x

	def loss(self, x, y, train_flg = False):
		m = x.shape[1]
		a = self.predict(x, train_flg)

		return self.lastLayer.forward(a, y)

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

		grads['W1'] = self.layers['Conv1'].dW
		grads['b1'] = self.layers['Conv1'].db
		grads['W2'] = self.layers['Conv2'].dW
		grads['b2'] = self.layers['Conv2'].db
		grads['W3'] = self.layers['Affine1'].dW
		grads['b3'] = self.layers['Affine1'].db
		grads['W4'] = self.layers['Affine2'].dW
		grads['b4'] = self.layers['Affine2'].db

		return grads, loss

	def set_unlearnable_params(self):
		pass

	def accuracy(self, x, y):
		a = self.predict(x)
		a_label = np.argmax(a, axis = 0)
		y_label = np.argmax(y, axis = 0)
		accuracy = np.mean(a_label == y_label) * 100

		return accuracy

	def returnParams(self):
		return self.params.copy(), self.unlearnable_params.copy()