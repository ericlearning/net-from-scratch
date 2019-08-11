from HypeNet.Core.NetworkFunctions import *
from HypeNet.Core.utils import *

class Linear:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = identity(x)
		return x

	def backward(self, dout):
		dx = d_identity(self.x) * dout
		return dx

class Sigmoid:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = sigmoid(self.x)
		return out

	def backward(self, dout):
		dx = d_sigmoid(self.x) * dout
		return dx

class Tanh:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = tanh(self.x)
		return out

	def backward(self, dout):
		dx = d_tanh(self.x) * dout
		return dx

class Relu:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = relu(self.x)
		return out

	def backward(self, dout):
		dx = d_relu(self.x) * dout
		return dx

class LeakyRelu:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = leakyrelu(self.x)
		return out

	def backward(self, dout):
		dx = d_leakyrelu(self.x) * dout
		return dx

class Elu:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = elu(self.x)
		return out

	def backward(self, dout):
		dx = d_elu(self.x) * dout
		return dx

class Selu:
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		out = selu(self.x)
		return out

	def backward(self, dout):
		dx = d_selu(self.x) * dout
		return dx

class MeanSquaredError:
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self, x, y):
		self.x = x
		self.y = y
		cost = mean_squared_error(self.x, self.y)
		return cost

	def backward(self, dout = 1):
		dx = (self.x - self.y) * dout
		return dx

class SoftmaxWithCrossEntropy:
	def __init__(self):
		self.a = None
		self.y = None

	def forward(self, x, y):
		self.a = softmax(x)
		self.y = y
		
		cost = cross_entropy_error(self.a, self.y)
		return cost

	def backward(self, dout = 1):
		dx = (self.a - self.y) * dout
		return dx

class Dropout:
	def __init__(self, keep_prob = 0.9):
		self.keep_prob = keep_prob
		self.mask = None

	def forward(self, x, train_flg = False):
		if(train_flg == True):
			self.mask = np.random.rand(*x.shape) < self.keep_prob
			return x * self.mask / self.keep_prob
		else:
			return x

	def backward(self, dout, calculate_dx = True):
		if(calculate_dx == True):
			dx = dout * self.mask / self.keep_prob
			return dx

#Convolution Finished
class Convolution:
	def __init__(self, W, b, stride = 1, pad = 0):
		self.W = W 									#(FN, C, FH, FW)
		self.b = b 									#(FN, 1)
		self.stride = stride
		self.pad = pad
		self.x = None
		self.X_col = None
		self.dW = None
		self.db = None

	def forward(self, x):
		FN, C, FH, FW = self.W.shape
		self.x = x
		XN, XC, XH, XW = self.x.shape

		OH = (XH + 2 * self.pad - FH) // self.stride + 1
		OW = (XW + 2 * self.pad - FW) // self.stride + 1
		self.X_col = im2col(self.x, FH, FW, self.stride, self.pad)		#(FH * FW * C, OH * OW * XN)	C = XC
		W_col = self.W.reshape(FN, -1)								#(FN, FH * FW * C)
		out = W_col @ self.X_col + self.b 							#(FN, OH * OW * XN)
		out = out.reshape(FN, OH, OW, XN).transpose(3, 0, 1, 2)			#(XN, FN, OH, OW)

		return out

	def backward(self, dout, calculate_dx = True):						#(XN, FN, OH, OW)
		FN, C, FH, FW = self.W.shape
		dout_flat = dout.transpose(1, 2, 3, 0)							#(FN, OH, OW, XN)
		dout_flat = dout_flat.reshape(FN, -1)							#(FN, OH * OW * XN)
		self.dW = dout_flat @ self.X_col.T								#(FN, FH * FW * C) = (FN, OH * OW * XN) * (OH * OW * XN, FH * FW * C)
		self.dW = self.dW.reshape(*self.W.shape)
		self.db = np.sum(dout, axis = (0, 2, 3))
		self.db = self.db.reshape(*self.b.shape)

		if(calculate_dx == True):
			W_col = self.W.reshape(FN, -1)
			dx_col = W_col.T @ dout_flat
			dx = col2im(dx_col, self.x.shape, FH, FW, self.stride, self.pad)
			return dx

#Pooling Finished
class Pooling:
	def __init__(self, pool_h, pool_w, stride = 2, pad = 0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad
		self.x = None
		self.X_col = None
		self.X_argmax = None

	def forward(self, x):
		self.x = x
		XN, XC, XH, XW = self.x.shape

		OH = (XH + 2 * self.pad - self.pool_h) // self.stride + 1
		OW = (XW + 2 * self.pad - self.pool_w) // self.stride + 1

		x_reshaped = self.x.reshape(XN * XC, 1, XH, XW)													#(XN * XC, 1, XH, XW)
		self.X_col = im2col(x_reshaped, self.pool_h, self.pool_w, self.stride, self.pad)				#(pool_h * pool_w * 1, OH * OW * XN * XC)
		self.X_argmax = np.argmax(self.X_col, axis = 0)													#(, OH * OW * XN * XC)
		X_max = self.X_col[self.X_argmax, range(self.X_argmax.size)]								#(, OH * OW * XN * XC)

		out = X_max.reshape(OH, OW, XN, XC).transpose(2, 3, 0, 1)									#(XN, XC, OH, OW)
		return out

	def backward(self, dout, calculate_dx = True):
		if(calculate_dx == True):
			XN, XC, XH, XW = self.x.shape
			dX_col = np.zeros(shape = self.X_col.shape)
			dout_flat = dout.transpose(2, 3, 0, 1)														#(OH, OW, XN, XC)
			dout_flat = dout_flat.ravel()																#(, OH * OW * XN * XC)
			dX_col[self.X_argmax, range(self.X_argmax.size)] = dout_flat								#(pool_h * pool_w * 1, OH * OW * XN * XC)
			reshaped_shape = (XN * XC, 1, XH, XW)
			dX = col2im(dX_col, reshaped_shape, self.pool_h, self.pool_w, self.stride, self.pad)			#(XN, XC, XH, XW)
			#now, dX has the same shape with X
			return dX.reshape(*self.x.shape)
'''
class UpSampling:
	def __init__(self, pool_h, pool_w, max_locations, stride = 2, pad = 0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.max_locations = max_locations
		self.stride = stride
		self.pad = pad
		self.x = None

	def forward(self, x):
		self.x = x
		out = self.x.repeat(self.pool_h, axis = 2).repeat(self.pool_w, axis = 3)
		print(self.max_locations)
		return out

	def backward(self, dout, calculate_dx = True):
		if(calculate_dx == True):
			dout_col = im2col(dout, self.pool_h, self.pool_w, self.stride, self.pad)
			print(dout_col)
			return dout
'''
class BatchNormalization:
	def __init__(self, gamma, beta, momentum = 0.9, epsilon = 10e-7, running_mean = None, running_var = None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.epsilon = epsilon
		self.running_mean = running_mean
		self.running_var = running_var
		self.x_temp_shape = None

	def forward(self, x, train_flg = False):
		if(x.ndim == 4):								#Convolution in Previous Layer
			#(N, C, H, W)
			x = x.transpose(0, 2, 3, 1)					#(N, H, W, C)
			self.x_temp_shape = x.shape
			x = x.reshape(-1, x.shape[3])				#(N * H * W, C)
			x = x.T										#(C, N * H * W)

		self.x = x
		out = self.__forward(self.x, train_flg)
		return out

	def __forward(self, x, train_flg):
		if((self.running_mean is None) or (self.running_var is None)):
			self.running_mean = np.zeros((x.shape[0], 1))
			self.running_var = np.zeros((x.shape[0], 1))

		if(train_flg == True):
			mu = np.mean(x, axis = 1, keepdims = True)
			xc = x - mu
			xcsquared = xc**2
			var = np.mean(xcsquared, axis = 1, keepdims = True)
			std = np.sqrt(var + self.epsilon)
			istd = 1.0 / std
			xn = xc * istd
			
			self.mu = mu
			self.xc = xc
			self.xcsquared = xcsquared
			self.var = var
			self.std = std
			self.istd = istd
			self.xn = xn

			self.running_mean = (1 - self.momentum) * self.mu + (self.momentum) * self.running_mean
			self.running_var = (1 - self.momentum) * self.var + (self.momentum) * self.running_var

		else:
			xc = x - self.running_mean
			std = np.sqrt(self.running_var + self.epsilon)
			xn = xc / std

		out = xn * self.gamma + self.beta
		return out

	def backward(self, dout, calculate_dx = True):
		dx = self.__backward(dout, calculate_dx)
		if(self.x_temp_shape == None):
			dx = dx.reshape(*self.x.shape)
		else:
			dx = dx.T 								#(N * H * W, C)
			dx = dx.reshape(self.x_temp_shape)		#(N, H, W, C)
			dx = dx.transpose(0, 3, 1, 2)

		return dx

	def __backward(self, dout, calculate_dx):
		m = dout.shape[1]
		self.dbeta = np.sum(dout, axis = 1, keepdims = True)
		self.dgamma = np.sum(dout * self.xn, axis = 1, keepdims = True)

		if(calculate_dx == True):
			dxn = dout * self.gamma
			distd = np.sum(dxn * self.xc, axis = 1, keepdims = True)
			dstd = -1.0 / (self.std ** 2) * distd
			dvar = 0.5 * 1 / (self.std) * dstd
			dxcsquared = (1.0 / m) * np.ones(shape = self.x.shape) * dvar
			dxc_2 = 2.0 * dxcsquared * self.xc
			dxc_1 = dxn * self.istd
			dxc = dxc_1 + dxc_2
			dmu = -np.sum(dxc, axis = 1, keepdims = True)
			dx_2 = (1.0 / m) * np.ones(shape = self.x.shape) * dmu
			dx_1 = dxc
			dx = dx_1 + dx_2
			return dx

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None
		self.x_original_shape = None

	def forward(self, x):
		if(x.ndim == 4):								#Convolution in Previous Layer
			self.x_original_shape = x.shape 			#(N, C, H, W)
			x = x.transpose(1, 2, 3, 0)					#(C, H, W, N)
			x = x.reshape(-1, x.shape[3])				#(C * H * W, N)

		self.x = x
		out = self.W @ self.x + self.b
		return out

	def backward(self, dout, calculate_dx = True):
		m = dout.shape[1]
		self.dW = dout @ self.x.T / m
		self.db = np.sum(dout, axis = 1, keepdims = True) / m
		
		if(calculate_dx == True):
			dx = self.W.T @ dout
			if(self.x_original_shape != None):
				dx = dx.T 								#(N, C * H * W)
				dx = dx.reshape(*self.x_original_shape)	#(N, C, H, W)
			return dx