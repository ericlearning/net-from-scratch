import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from HypeNet.Networks.FCNN_MSE import FCNN_MSE
import os, math

def generate_img(network, latent_value):
	X = np.linspace(-w / mean, w / mean, w)
	Y = np.linspace(-h / mean, h / mean, h)
	X, Y = np.meshgrid(X, Y)
	X = X.ravel().reshape(1, -1)
	Y = Y.ravel().reshape(1, -1)

	R = np.sqrt(X ** 2 + Y ** 2)
	Z = np.repeat(latent_value, X.shape[1]).reshape(1, -1)

	XYRZ = np.concatenate((X, Y, R, Z), axis = 0)
	output_img = network.predict(XYRZ.astype('float64')).reshape(3, h, w).transpose(1, 2, 0)

	output_img = 255.0 * ((output_img - np.min(output_img)) / (np.max(output_img) - np.min(output_img)))
	output_img = output_img.astype(np.uint8)

	return output_img

def generate_img_grayscale(network, latent_value):
	X = np.linspace(-w / mean, w / mean, w)
	Y = np.linspace(-h / mean, h / mean, h)
	X, Y = np.meshgrid(X, Y)
	X = X.ravel().reshape(1, -1)
	Y = Y.ravel().reshape(1, -1)

	R = np.sqrt(X ** 2 + Y ** 2)
	Z = np.repeat(latent_value, X.shape[1]).reshape(1, -1)

	XYRZ = np.concatenate((X, Y, R, Z), axis = 0)
	output_img = network.predict(XYRZ.astype('float64')).reshape(h, w)

	output_img = 255.0 * ((output_img - np.min(output_img)) / (np.max(output_img) - np.min(output_img)))
	output_img = output_img.astype(np.uint8)

	return output_img

h, w = 1600, 2560
mean = (h + w) / 2.0

layer_num = [32, 32, 32, 32, 32, 32, 32, 32]
activations = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
network = FCNN_MSE(4, layer_num, 3, activations, weight_init_std = 1.0)

output_imgs = []
latent_value = 0.5
fig = plt.figure('Pattern Produced', figsize = (8, 8))

output_img = generate_img(network, latent_value)

#plt.imshow(output_img, cmap = plt.get_cmap('gray'))
#plt.imsave('cool.png', output_img, cmap = plt.get_cmap('gray'))

plt.imshow(output_img)
plt.imsave('cool2.png', output_img)

plt.show()