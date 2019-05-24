import numpy as np

class SGD:
	def __init__(self, lr = 0.01):
		self.lr = lr

	def update(self, params, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]

class Momentum:
	def __init__(self, lr = 0.01, beta = 0.9):
		self.lr = lr
		self.beta = beta
		self.v = None

	def update(self, params, grads):
		if(self.v == None):
			self.v = {}
			for key, val in params.items():
				self.v[key] = np.zeros(shape = val.shape)

		for key in params.keys():
			self.v[key] = self.beta * self.v[key] - self.lr * grads[key]
			params[key] += self.v[key]

class Nesterov:
	def __init__(self, lr = 0.01, beta = 0.9):
		self.lr = lr
		self.beta = beta
		self.v = None

	def update(self, params, grads):
		if(self.v == None):
			self.v = {}
			for key, val in params.items():
				self.v[key] = np.zeros(shape = val.shape)

		for key in params.keys():
			v_prev = self.v[key]
			self.v[key] = self.beta * v_prev - self.lr * grads[key]
			params[key] += -self.beta * v_prev + (1 + self.beta) * self.v[key]

class RmsProp:
	def __init__(self, lr = 0.01, beta = 0.9, epsilon = 1e-8):
		self.lr = lr
		self.beta = beta
		self.epsilon = epsilon
		self.s = None

	def update(self, params, grads):
		if(self.s == None):
			self.s = {}
			for key, val in params.items():
				self.s[key] = np.zeros(shape = val.shape)

		for key in params.keys():
			self.s[key] = self.beta * self.s[key] + (1 - self.beta) * (grads[key]**2)
			params[key] -= self.lr * (grads[key] / (np.sqrt(self.s[key]) + self.epsilon))

class Adam:
	def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.999, t = 2, epsilon = 1e-8):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.v = None
		self.s = None
		self.v_corrected = None
		self.s_corrected = None
		self.t = t
		self.epsilon = epsilon

	def update(self, params, grads):
		if(self.s == None or self.v == None):
			self.v = {}
			self.s = {}
			self.v_corrected = {}
			self.s_corrected = {}
			for key, val in params.items():
				self.v[key] = np.zeros(shape = val.shape)
				self.s[key] = np.zeros(shape = val.shape)
				self.v_corrected[key] = np.zeros(shape = val.shape)
				self.s_corrected[key] = np.zeros(shape = val.shape)

		for key in params.keys():
			self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grads[key]
			self.v_corrected[key] = self.v[key] / (1 - self.beta1 ** self.t)

			self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grads[key] ** 2)
			self.s_corrected[key] = self.s[key] / (1 - self.beta2 ** self.t)

			params[key] -= self.lr * (self.v_corrected[key] / (np.sqrt(self.s_corrected[key]) + self.epsilon))

class Nadam:
	def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.999, t = 2, epsilon = 1e-8):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.v = None
		self.s = None
		self.v_corrected = None
		self.s_corrected = None
		self.t = t
		self.epsilon = epsilon

	def update(self, params, grads):
		if(self.s == None or self.v == None):
			self.v = {}
			self.s = {}
			self.v_corrected = {}
			self.s_corrected = {}
			for key, val in params.items():
				self.v[key] = np.zeros(shape = val.shape)
				self.s[key] = np.zeros(shape = val.shape)
				self.v_corrected[key] = np.zeros(shape = val.shape)
				self.s_corrected[key] = np.zeros(shape = val.shape)

		for key in params.keys():
			self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grads[key]
			self.v_corrected[key] = self.v[key] / (1 - self.beta1 ** self.t)

			self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grads[key] ** 2)
			self.s_corrected[key] = self.s[key] / (1 - self.beta2 ** self.t)

			denominator = np.sqrt(self.s_corrected[key]) + self.epsilon
			numerator = self.beta1 * self.v_corrected[key] + ((1 - self.beta1) * grads[key] / (1 - self.beta1 ** self.t))

			params[key] -= self.lr * (numerator / denominator)

class AmsGrad:
	def __init__(self, lr = 0.01, beta1 = 0.9, beta2 = 0.999, t = 2, epsilon = 1e-8):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.v = None
		self.s = None
		self.v_corrected = None
		self.s_corrected = None
		self.t = t
		self.epsilon = epsilon

	def update(self, params, grads):
		if(self.s == None or self.v == None):
			self.v = {}
			self.s = {}
			self.v_corrected = {}
			self.s_corrected = {}
			for key, val in params.items():
				self.v[key] = np.zeros(shape = val.shape)
				self.s[key] = np.zeros(shape = val.shape)
				self.v_corrected[key] = np.zeros(shape = val.shape)
				self.s_corrected[key] = np.zeros(shape = val.shape)

		for key in params.keys():
			self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grads[key]
			self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grads[key] ** 2)

			self.s_corrected[key] = np.maximum(self.s_corrected[key], self.s[key])
				
			params[key] -= self.lr * (self.v[key] / (np.sqrt(self.s_corrected[key]) + self.epsilon))
