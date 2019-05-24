import numpy as np

def sigmoid(z):
	a = 1/(1+np.exp(-z))
	return a

def d_sigmoid(z):
	a = sigmoid(z) * (1 - sigmoid(z))
	return a

def tanh(z):
	a = np.tanh(z)
	return a

def d_tanh(z):
	a = 1 - tanh(z)**2
	return a

def relu(z):
	a = np.maximum(z, 0)
	return a

def leakyrelu(z, alpha = 0.3):
	a = np.maximum(z, alpha * z)
	return a
	
def d_leakyrelu(z, alpha = 0.3):
	a = np.zeros(shape = z.shape)
	a[z<0] = alpha
	a[z>=0] = 1
	return a

def d_relu(z):
	a = np.zeros(shape = z.shape)
	a[z<0] = 0
	a[z>=0] = 1
	return a

def elu(z, alpha = 1.0):
	a = alpha * (np.exp(z) - 1)
	a[z>=0] = z[z>=0]
	return a

def d_elu(z, alpha = 1.0):
	a = elu(z) + alpha
	a[z>=0] = 1
	return a

def selu(z, alpha = 1.67326, gamma = 1.0507):
	return elu(z, alpha) * gamma

def d_selu(z, alpha = 1.67326, gamma = 1.0507):
	return d_elu(z, alpha) * gamma

def identity(z):
	return z

def d_identity(z):
	return 1

def mean_squared_error(A, Y):
	m = A.shape[1]
	cost = np.square(A - Y).sum() / (2.0 * m)
	return cost

def cross_entropy_error(A, Y):
	m = A.shape[1]
	cost = np.sum(Y * np.log(A)) * (-1 / m)
	return cost

def softmax(X):
	X = X - np.max(X, axis = 0)
	exp_X = np.exp(X)
	return exp_X / np.sum(exp_X, axis = 0)