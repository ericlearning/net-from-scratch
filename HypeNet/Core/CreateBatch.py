import math
from HypeNet.Core.utils import *

def createBatch(X, Y, mini_batch_size):
	if(X.ndim == 2):
		m = X.shape[1]
		mini_batches = []

		shuffled_X, shuffled_Y = shuffle_data(X, Y)

		complete_mini_batch_num = math.floor(m / mini_batch_size)

		for k in range(0, complete_mini_batch_num):
			mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
			mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		if(m % mini_batch_size != 0):
			mini_batch_X = shuffled_X[:, complete_mini_batch_num * mini_batch_size : m]
			mini_batch_Y = shuffled_Y[:, complete_mini_batch_num * mini_batch_size : m]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		return mini_batches

	elif(X.ndim == 4):
		m = X.shape[0]
		mini_batches = []

		shuffled_X, shuffled_Y = shuffle_data(X, Y)

		complete_mini_batch_num = math.floor(m / mini_batch_size)

		for k in range(0, complete_mini_batch_num):
			mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size, :, :, :]
			mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		if(m % mini_batch_size != 0):
			mini_batch_X = shuffled_X[complete_mini_batch_num * mini_batch_size : m, :, :, :]
			mini_batch_Y = shuffled_Y[:, complete_mini_batch_num * mini_batch_size : m]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		return mini_batches