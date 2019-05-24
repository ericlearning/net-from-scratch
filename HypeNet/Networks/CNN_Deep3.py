import os
import numpy as np
from collections import OrderedDict
from HypeNet.Core.NetworkLayers import *
from HypeNet.Core.NetworkFunctions import cross_entropy_error
from HypeNet.Core.utils import calculateConvOutputSize

'''
[Architecture]
	Input -> Conv(32,3,3) -> Relu -> Conv(32,3,3) -> Relu -> Pool
		  -> Conv(64,3,3) -> Relu -> Conv(64,3,3) -> Relu -> Pool
		  -> Conv(128,3,3) -> Relu -> Conv(128,3,3) -> Relu -> Pool
		  -> Conv(256,3,3) -> Relu -> Conv(256,3,3) -> Relu -> Pool
		  -> Conv(512,3,3) -> Relu -> Conv(512,3,3) -> Relu -> Pool
		  -> Affine -> Relu -> Dropout
		  -> Affine -> Relu -> Dropout
		  -> Affine -> Dropout -> Softmax
'''
class CNN_Deep3:
	def __init__(self, input_dim = (3, 256, 256), conv_param1 = {'filter_num' : 32, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param2 = {'filter_num' : 32, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param3 = {'filter_num' : 64, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param4 = {'filter_num' : 64, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param5 = {'filter_num' : 128, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param6 = {'filter_num' : 128, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param7 = {'filter_num' : 256, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param8 = {'filter_num' : 256, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param9 = {'filter_num' : 512, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, conv_param10 = {'filter_num' : 512, 'filter_height' : 3, 'filter_width' : 3, 'pad' : 1, 'stride' : 1}
												, hidden_size_1 = 1024, hidden_size_2 = 1024, output_size = 2):

		self.input_dim = input_dim
		self.conv_param1 = conv_param1
		self.conv_param2 = conv_param2
		self.conv_param3 = conv_param3
		self.conv_param4 = conv_param4
		self.conv_param5 = conv_param5
		self.conv_param6 = conv_param6
		self.conv_param7 = conv_param7
		self.conv_param8 = conv_param8
		self.conv_param9 = conv_param9
		self.conv_param10 = conv_param10
		self.hidden_size_1 = hidden_size_1
		self.hidden_size_2 = hidden_size_2
		self.output_size = output_size

		self.unlearnable_params = {}
		self.params = {}
		self.paramInit()

		self.layers = OrderedDict()
		self.layerInit()

	def calculate_final_conv_output_size(self):
		input_H, input_W = self.input_dim[1], self.input_dim[2]

		conv1_output_H = calculateConvOutputSize(input_H, self.conv_param1['pad'], self.conv_param1['filter_height'], self.conv_param1['stride'])
		conv2_output_H = calculateConvOutputSize(conv1_output_H, self.conv_param2['pad'], self.conv_param2['filter_height'], self.conv_param2['stride'])
		pool1_output_H = conv2_output_H // 2
		conv3_output_H = calculateConvOutputSize(pool1_output_H, self.conv_param3['pad'], self.conv_param3['filter_height'], self.conv_param3['stride'])
		conv4_output_H = calculateConvOutputSize(conv3_output_H, self.conv_param4['pad'], self.conv_param4['filter_height'], self.conv_param4['stride'])
		pool2_output_H = conv4_output_H // 2
		conv5_output_H = calculateConvOutputSize(pool2_output_H, self.conv_param5['pad'], self.conv_param5['filter_height'], self.conv_param5['stride'])
		conv6_output_H = calculateConvOutputSize(conv5_output_H, self.conv_param6['pad'], self.conv_param6['filter_height'], self.conv_param6['stride'])
		pool3_output_H = conv6_output_H // 2
		conv7_output_H = calculateConvOutputSize(pool3_output_H, self.conv_param7['pad'], self.conv_param7['filter_height'], self.conv_param7['stride'])
		conv8_output_H = calculateConvOutputSize(conv7_output_H, self.conv_param8['pad'], self.conv_param8['filter_height'], self.conv_param8['stride'])
		pool4_output_H = conv8_output_H // 2
		conv9_output_H = calculateConvOutputSize(pool4_output_H, self.conv_param9['pad'], self.conv_param9['filter_height'], self.conv_param9['stride'])
		conv10_output_H = calculateConvOutputSize(conv9_output_H, self.conv_param10['pad'], self.conv_param10['filter_height'], self.conv_param10['stride'])
		pool5_output_H = conv10_output_H // 2

		conv1_output_W = calculateConvOutputSize(input_W, self.conv_param1['pad'], self.conv_param1['filter_width'], self.conv_param1['stride'])
		conv2_output_W = calculateConvOutputSize(conv1_output_W, self.conv_param2['pad'], self.conv_param2['filter_width'], self.conv_param2['stride'])
		pool1_output_W = conv2_output_W // 2
		conv3_output_W = calculateConvOutputSize(pool1_output_W, self.conv_param3['pad'], self.conv_param3['filter_width'], self.conv_param3['stride'])
		conv4_output_W = calculateConvOutputSize(conv3_output_W, self.conv_param4['pad'], self.conv_param4['filter_width'], self.conv_param4['stride'])
		pool2_output_W = conv4_output_W // 2
		conv5_output_W = calculateConvOutputSize(pool2_output_W, self.conv_param5['pad'], self.conv_param5['filter_width'], self.conv_param5['stride'])
		conv6_output_W = calculateConvOutputSize(conv5_output_W, self.conv_param6['pad'], self.conv_param6['filter_width'], self.conv_param6['stride'])
		pool3_output_W = conv6_output_W // 2
		conv7_output_W = calculateConvOutputSize(pool3_output_W, self.conv_param7['pad'], self.conv_param7['filter_width'], self.conv_param7['stride'])
		conv8_output_W = calculateConvOutputSize(conv7_output_W, self.conv_param8['pad'], self.conv_param8['filter_width'], self.conv_param8['stride'])
		pool4_output_W = conv8_output_W // 2
		conv9_output_W = calculateConvOutputSize(pool4_output_W, self.conv_param9['pad'], self.conv_param9['filter_height'], self.conv_param9['stride'])
		conv10_output_W = calculateConvOutputSize(conv9_output_W, self.conv_param10['pad'], self.conv_param10['filter_height'], self.conv_param10['stride'])
		pool5_output_W = conv10_output_W // 2

		final_conv_output_H = pool5_output_H
		final_conv_output_W = pool5_output_W

		return final_conv_output_H, final_conv_output_W

	def paramInit(self):
		(final_conv_output_H, final_conv_output_W), final_conv_output_C = self.calculate_final_conv_output_size(), self.conv_param10['filter_num']		#(8, 8, 512)
		
		prev_connect_num = [self.input_dim[0] * self.conv_param1['filter_height'] * self.conv_param1['filter_width'], 
							self.conv_param1['filter_num'] * self.conv_param2['filter_height'] * self.conv_param2['filter_width'], 
							self.conv_param2['filter_num'] * self.conv_param3['filter_height'] * self.conv_param3['filter_width'], 
							self.conv_param3['filter_num'] * self.conv_param4['filter_height'] * self.conv_param4['filter_width'], 
							self.conv_param4['filter_num'] * self.conv_param5['filter_height'] * self.conv_param5['filter_width'], 
							self.conv_param5['filter_num'] * self.conv_param6['filter_height'] * self.conv_param6['filter_width'],
							self.conv_param6['filter_num'] * self.conv_param7['filter_height'] * self.conv_param7['filter_width'],
							self.conv_param7['filter_num'] * self.conv_param8['filter_height'] * self.conv_param8['filter_width'], 
							self.conv_param8['filter_num'] * self.conv_param9['filter_height'] * self.conv_param9['filter_width'], 
							self.conv_param9['filter_num'] * self.conv_param10['filter_height'] * self.conv_param10['filter_width'], 
							final_conv_output_H * final_conv_output_W * final_conv_output_C, self.hidden_size_1, self.hidden_size_2]
							
		print(prev_connect_num)
		print((final_conv_output_H, final_conv_output_W, final_conv_output_C))

		self.params['W1'] = np.random.randn(self.conv_param1['filter_num'], self.input_dim[0], self.conv_param1['filter_height'], self.conv_param1['filter_width']) * np.sqrt(2.0 / prev_connect_num[0])
		self.params['b1'] = np.zeros((self.conv_param1['filter_num'], 1))
		self.params['W2'] = np.random.randn(self.conv_param2['filter_num'], self.conv_param1['filter_num'], self.conv_param2['filter_height'], self.conv_param2['filter_width']) * np.sqrt(2.0 / prev_connect_num[1])
		self.params['b2'] = np.zeros((self.conv_param2['filter_num'], 1))
		self.params['W3'] = np.random.randn(self.conv_param3['filter_num'], self.conv_param2['filter_num'], self.conv_param3['filter_height'], self.conv_param3['filter_width']) * np.sqrt(2.0 / prev_connect_num[2])
		self.params['b3'] = np.zeros((self.conv_param3['filter_num'], 1))
		self.params['W4'] = np.random.randn(self.conv_param4['filter_num'], self.conv_param3['filter_num'], self.conv_param4['filter_height'], self.conv_param4['filter_width']) * np.sqrt(2.0 / prev_connect_num[3])
		self.params['b4'] = np.zeros((self.conv_param4['filter_num'], 1))
		self.params['W5'] = np.random.randn(self.conv_param5['filter_num'], self.conv_param4['filter_num'], self.conv_param5['filter_height'], self.conv_param5['filter_width']) * np.sqrt(2.0 / prev_connect_num[4])
		self.params['b5'] = np.zeros((self.conv_param5['filter_num'], 1))
		self.params['W6'] = np.random.randn(self.conv_param6['filter_num'], self.conv_param5['filter_num'], self.conv_param6['filter_height'], self.conv_param6['filter_width']) * np.sqrt(2.0 / prev_connect_num[5])
		self.params['b6'] = np.zeros((self.conv_param6['filter_num'], 1))
		self.params['W7'] = np.random.randn(self.conv_param7['filter_num'], self.conv_param6['filter_num'], self.conv_param7['filter_height'], self.conv_param7['filter_width']) * np.sqrt(2.0 / prev_connect_num[6])
		self.params['b7'] = np.zeros((self.conv_param7['filter_num'], 1))
		self.params['W8'] = np.random.randn(self.conv_param8['filter_num'], self.conv_param7['filter_num'], self.conv_param8['filter_height'], self.conv_param8['filter_width']) * np.sqrt(2.0 / prev_connect_num[7])
		self.params['b8'] = np.zeros((self.conv_param8['filter_num'], 1))
		self.params['W9'] = np.random.randn(self.conv_param9['filter_num'], self.conv_param8['filter_num'], self.conv_param9['filter_height'], self.conv_param9['filter_width']) * np.sqrt(2.0 / prev_connect_num[8])
		self.params['b9'] = np.zeros((self.conv_param9['filter_num'], 1))
		self.params['W10'] = np.random.randn(self.conv_param10['filter_num'], self.conv_param9['filter_num'], self.conv_param10['filter_height'], self.conv_param10['filter_width']) * np.sqrt(2.0 / prev_connect_num[9])
		self.params['b10'] = np.zeros((self.conv_param10['filter_num'], 1))

		self.params['W11'] = np.random.randn(prev_connect_num[11], prev_connect_num[10]) * np.sqrt(2.0 / prev_connect_num[10])
		self.params['b11'] = np.zeros((prev_connect_num[11], 1))
		self.params['W12'] = np.random.randn(prev_connect_num[12], prev_connect_num[11]) * np.sqrt(2.0 / prev_connect_num[11])
		self.params['b12'] = np.zeros((prev_connect_num[12], 1))
		self.params['W13'] = np.random.randn(self.output_size, prev_connect_num[12]) * np.sqrt(2.0 / prev_connect_num[12])
		self.params['b13'] = np.zeros((self.output_size, 1))

	def paramInit_pretrained(self, pretrained_params, unlearnable_pretrained_params):
		for i in range(13):
			self.params['W' + str(i+1)] = pretrained_params['W' + str(i+1)]
			self.params['b' + str(i+1)] = pretrained_params['b' + str(i+1)]

	def layerInit(self):
		self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], self.conv_param1['stride'], self.conv_param1['pad'])
		self.layers['Relu1'] = Relu()
		self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], self.conv_param2['stride'], self.conv_param2['pad'])
		self.layers['Relu2'] = Relu()
		self.layers['Pool1'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)

		self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], self.conv_param3['stride'], self.conv_param3['pad'])
		self.layers['Relu3'] = Relu()
		self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], self.conv_param4['stride'], self.conv_param4['pad'])
		self.layers['Relu4'] = Relu()
		self.layers['Pool2'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)

		self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], self.conv_param5['stride'], self.conv_param5['pad'])
		self.layers['Relu5'] = Relu()
		self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'], self.conv_param6['stride'], self.conv_param6['pad'])
		self.layers['Relu6'] = Relu()
		self.layers['Pool3'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)

		self.layers['Conv7'] = Convolution(self.params['W7'], self.params['b7'], self.conv_param7['stride'], self.conv_param7['pad'])
		self.layers['Relu7'] = Relu()
		self.layers['Conv8'] = Convolution(self.params['W8'], self.params['b8'], self.conv_param8['stride'], self.conv_param8['pad'])
		self.layers['Relu8'] = Relu()
		self.layers['Pool4'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)

		self.layers['Conv9'] = Convolution(self.params['W9'], self.params['b9'], self.conv_param9['stride'], self.conv_param9['pad'])
		self.layers['Relu9'] = Relu()
		self.layers['Conv10'] = Convolution(self.params['W10'], self.params['b10'], self.conv_param10['stride'], self.conv_param10['pad'])
		self.layers['Relu10'] = Relu()
		self.layers['Pool5'] = Pooling(pool_h = 2, pool_w = 2, stride = 2, pad = 0)

		self.layers['Affine1'] = Affine(self.params['W11'], self.params['b11'])
		self.layers['Relu11'] = Relu()
		self.layers['Dropout1'] = Dropout(0.5)

		self.layers['Affine2'] = Affine(self.params['W12'], self.params['b12'])
		self.layers['Relu12'] = Relu()
		self.layers['Dropout2'] = Dropout(0.5)

		self.layers['Affine3'] = Affine(self.params['W13'], self.params['b13'])
		self.layers['Dropout3'] = Dropout(0.5)
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
		grads['W3'] = self.layers['Conv3'].dW
		grads['b3'] = self.layers['Conv3'].db
		grads['W4'] = self.layers['Conv4'].dW
		grads['b4'] = self.layers['Conv4'].db
		grads['W5'] = self.layers['Conv5'].dW
		grads['b5'] = self.layers['Conv5'].db
		grads['W6'] = self.layers['Conv6'].dW
		grads['b6'] = self.layers['Conv6'].db
		grads['W7'] = self.layers['Conv7'].dW
		grads['b7'] = self.layers['Conv7'].db
		grads['W8'] = self.layers['Conv8'].dW
		grads['b8'] = self.layers['Conv8'].db
		grads['W9'] = self.layers['Conv9'].dW
		grads['b9'] = self.layers['Conv9'].db
		grads['W10'] = self.layers['Conv10'].dW
		grads['b10'] = self.layers['Conv10'].db

		grads['W11'] = self.layers['Affine1'].dW
		grads['b11'] = self.layers['Affine1'].db
		grads['W12'] = self.layers['Affine2'].dW
		grads['b12'] = self.layers['Affine2'].db
		grads['W13'] = self.layers['Affine3'].dW
		grads['b13'] = self.layers['Affine3'].db

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