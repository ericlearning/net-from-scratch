import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os

def calculateConvOutputSize(I, P, F, S):
	O = (I + 2 * P - F) // S + 1
	return O

def im2col(X, FH, FW, S = 1, P = 0):
	N, C, H, W = X.shape
	OH = (H + 2 * P - FH) // S + 1
	OW = (W + 2 * P - FW) // S + 1

	X_padded = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode = 'constant')

	k = np.repeat(np.arange(C), FH * FW).reshape(-1, 1)
	i1 = np.tile(np.repeat(np.arange(FH), FW), C).reshape(-1, 1)
	i2 = np.repeat(np.arange(OH) * S, OW).reshape(1, -1)
	i = i1 + i2
	j1 = np.tile(np.arange(FW), FH * C).reshape(-1, 1)
	j2 = np.tile(np.arange(OW) * S, OH).reshape(1, -1)
	j = j1 + j2

	X_col = X_padded[:, k, i, j].transpose(1, 2, 0).reshape(FH * FW * C, OH * OW * N)
	return X_col

def col2im(col, X_shape, FH, FW, S = 1, P = 0):
	N, C, H, W = X_shape
	H_padded = H + 2 * P
	W_padded = W + 2 * P
	OH = (H + 2 * P - FH) // S + 1
	OW = (W + 2 * P - FW) // S + 1

	X_padded = np.zeros((N, C, H_padded, W_padded))

	k = np.repeat(np.arange(C), FH * FW).reshape(-1, 1)
	i1 = np.tile(np.repeat(np.arange(FH), FW), C).reshape(-1, 1)
	i2 = np.repeat(np.arange(OH) * S, OW).reshape(1, -1)
	i = i1 + i2
	j1 = np.tile(np.arange(FW), FH * C).reshape(-1, 1)
	j2 = np.tile(np.arange(OW) * S, OH).reshape(1, -1)
	j = j1 + j2

	col_reshape = col.reshape(C * FH * FW, OH * OW, N)

	col_reshape = col_reshape.transpose(2, 0, 1)	
	
	np.add.at(X_padded, (slice(None), k, i, j), col_reshape)

	if(P == 0):
		return X_padded
	else:
		return X_padded[:, :, P:-P, P:-P]

def lerp(a0, a1, t):
	a = a0 + (a1 - a0) * t
	return a

def networkSaver(network, DIR):
	params, unlearnable_params = network.returnParams()
	for key, val in params.items():
		np.save(DIR + 'learnable_' + key + '.npy', val)
	for key, val in unlearnable_params.items():
		np.save(DIR + 'unlearnable_' + key + '.npy', val)

def networkLoader(network, DIR):
	params = {}
	unlearnable_params = {}

	for cur_file in os.listdir(DIR):
		if('unlearnable' in cur_file):
			cur_unlearnable_params = np.load(DIR + cur_file)
			unlearnable_params[cur_file[:-4].partition('unlearnable_')[2]] = cur_unlearnable_params
		elif('learnable' in cur_file):
			cur_learnable_params = np.load(DIR + cur_file)
			params[cur_file[:-4].partition('learnable_')[2]] = cur_learnable_params
			
	network.paramInit_pretrained(params, unlearnable_params)
	network.layerInit()

def categorical(X, category_num):
	X_categorical = np.zeros((category_num, X.shape[1]))
	X_ravel = np.ravel(X)
	X_arange = np.arange(X.shape[1])
	X_categorical[X_ravel, X_arange] = 1
	
	return X_categorical

def shuffle_data(X, Y):
	if(X.ndim == 2):
		perm = np.random.permutation(X.shape[1])
		X_shuffled = X.T[perm].T
		Y_shuffled = Y.T[perm].T
		return X_shuffled, Y_shuffled

	elif(X.ndim == 4):
		perm = np.random.permutation(X.shape[0])
		X_shuffled = X[perm]
		Y_shuffled = Y.T[perm].T
		return X_shuffled, Y_shuffled

def data_augmentation(X, Y, rotate = 15.0, flipHor = False, image_size = None):
	if(X.ndim == 2):
		m = X.shape[1]
		if(rotate != False):
			X_rotate = []
			Y_rotate = []
		if(flipHor == True):
			X_flipHor = []
			Y_flipHor = []

		for i in range(m):
			if(rotate != False):
				cur_X_rotate1 = ndimage.interpolation.rotate(X.T[i].reshape(image_size), angle = rotate, reshape = False)
				cur_X_rotate2 = ndimage.interpolation.rotate(X.T[i].reshape(image_size), angle = -rotate, reshape = False)
				X_rotate.append(cur_X_rotate1)
				X_rotate.append(cur_X_rotate2)
				Y_rotate.append(Y.T[i].T)
				Y_rotate.append(Y.T[i].T)

			if(flipHor == True):
				cur_X_flipHor = np.flip(X.T[i].reshape(image_size), axis = 1)
				X_flipHor.append(cur_X_flipHor)
				Y_flipHor.append(Y.T[i].T)

		if(rotate != False):
			X_rotate = np.asarray(X_rotate)
			X_rotate = X_rotate.transpose(1, 2, 0)
			X_rotate = X_rotate.reshape(X.shape[0], X.shape[1] * 2)
			Y_rotate = np.asarray(Y_rotate).T

		if(flipHor == True):
			X_flipHor = np.asarray(X_flipHor)
			X_flipHor = X_flipHor.transpose(1, 2, 0)
			X_flipHor = X_flipHor.reshape(X.shape[0], X.shape[1])
			Y_flipHor = np.asarray(Y_rotate).T

		X_augmented = np.copy(X)
		Y_augmented = np.copy(Y)

		if(rotate != False):
			X_augmented = np.concatenate((X_augmented, X_rotate), axis = 1)
			Y_augmented = np.concatenate((Y_augmented, Y_rotate), axis = 1)

		if(flipHor == True):
			X_augmented = np.concatenate((X_augmented, X_flipHor), axis = 1)
			Y_augmented = np.concatenate((Y_augmented, Y_flipHor), axis = 1)
		
		X_augmented, Y_augmented = shuffle_data(X_augmented, Y_augmented)

	return X_augmented, Y_augmented

def draw_border(network, X_train, Y_train, X_val, Y_val, h = 0.01, c_map = 'coolwarm', c_edge = 'black', show_plt = True):
	colormap = plt.get_cmap(c_map)
	#X[0] is the x axis of data X, and use it to get max & min x axis value
	#X[1] is the y axis of data X, and use it to get max & min y axis value
	x_min, x_max = min(X_train[0].min(), X_val[0].min()) - 1, max(X_train[0].max(), X_val[0].max()) + 1
	y_min, y_max = min(X_train[1].min(), X_val[1].min()) - 1, max(X_train[1].max(), X_val[1].max()) + 1

	#Y should be an one-hot vector, therefore turn it into a label.
	Y_train_label = np.argmax(Y_train, axis = 0)
	Y_val_label = np.argmax(Y_val, axis = 0)

	#create xx, yy. These are each x axis values, and y axis values of the final mesh. This also means that xx and yy has the same shape with the final mesh vector
	#these xx, yy are created with max & min x axis, y axis values with the distance of h
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	#first, ravel xx and yy to get the 1D vector each containing x axis, y axis values of the final mesh.
	#Then, put these all together to get a (2, data_num) vector, which is a vector holding all the positions of the final mesh points.
	Z = np.c_[xx.ravel(), yy.ravel()].T

	#this will be fed into the network, in order to predict the output value for all these mesh points.
	#the network will return a (category_num, data_num) vector, and the label that has the biggest values will be the final label of that point.
	#Also, the final vector didn't go through the final layers, softmax and loss functions. However, this does not matter because applying softmax does not change the small & large relationship at all.
	A = network.predict(Z)

	#to get the final label, use argmax. Also, reshape it back to the shape of xx, which is the shape of our final meshgrid
	A_label = np.argmax(A, axis = 0).reshape(xx.shape)

	#plot the meshgrid using A_label, containing the label for all the mesh points.
	plt.contourf(xx, yy, A_label, cmap = colormap)

	#set the x & y label of the plot
	plt.ylabel('x2')
	plt.xlabel('x1')

	#plot the scatter plot for data point X, and label it using Y_label,
	plt.scatter(X_train[0], X_train[1], c = Y_train_label, cmap = colormap, edgecolors = c_edge)
	plt.scatter(X_val[0], X_val[1], c = Y_val_label, cmap = colormap, edgecolors = c_edge, marker = '^')

	#finally show the plot(meshgrid + scatter plot), using plt.show()
	if(show_plt == True):
		plt.show()