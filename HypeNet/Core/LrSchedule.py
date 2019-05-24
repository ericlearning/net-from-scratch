import numpy as np
import math
import matplotlib.pyplot as plt

class LinearDecay:
	def __init__(self):
		self.lr = None

	def set_lr(self, lr):
		self.lr = lr

	def update_lr(self, iteration):
		return self.lr

class ExpDecay:
	def __init__(self, k):
		self.lr = None
		self.k = k

	def set_lr(self, lr):
		self.lr_init = lr

	def update_lr(self, iteration):
		self.lr = self.lr_init * math.exp(-self.k * iteration)
		return self.lr

class DivDecay:
	def __init__(self, k):
		self.lr = None
		self.k = k

	def set_lr(self, lr):
		self.lr_init = lr

	def update_lr(self, iteration):
		self.lr = self.lr_init / (1.0 + self.k * iteration)
		return self.lr

class Triangular:
	def __init__(self, stepsize, base_lr, max_lr):
		self.stepsize = stepsize
		self.base_lr = base_lr
		self.max_lr = max_lr
		self.lr = None

	def set_lr(self, lr):
		pass

	def update_lr(self, iteration):
		self.lr = (1 - (np.abs(((iteration / (self.stepsize * 2))-np.floor(iteration / (self.stepsize * 2))) - 0.5) * 2)) * (self.max_lr - self.base_lr) + self.base_lr
		return self.lr

class Triangular_2:
	def __init__(self, stepsize, base_lr, max_lr):
		self.stepsize = stepsize
		self.base_lr = base_lr
		self.max_lr = max_lr
		self.lr = None

	def set_lr(self, lr):
		pass

	def update_lr(self, iteration):
		if(iteration % (self.stepsize * 2) == 0 and 0 < iteration):
			self.max_lr = (self.max_lr + self.base_lr) / 2.0
		self.lr = (1 - (np.abs(((iteration / (self.stepsize * 2))-np.floor(iteration / (self.stepsize * 2))) - 0.5) * 2)) * (self.max_lr - self.base_lr) + self.base_lr
		return self.lr

class Triangular_exp:
	def __init__(self, stepsize, base_lr, max_lr, gamma = 0.99994):
		self.stepsize = stepsize
		self.base_lr = base_lr
		self.max_lr = max_lr
		self.gamma = gamma
		self.lr = None

	def set_lr(self, lr):
		pass

	def update_lr(self, iteration):
		if(iteration % (self.stepsize * 2) == 0):
			self.max_lr = self.max_lr * (self.gamma ** iteration)
		self.lr = (1 - (np.abs(((iteration / (self.stepsize * 2))-np.floor(iteration / (self.stepsize * 2))) - 0.5) * 2)) * (self.max_lr - self.base_lr) + self.base_lr
		return self.lr