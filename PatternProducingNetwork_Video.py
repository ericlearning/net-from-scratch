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

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
h, w = 1024, 1024
mean = (h + w) / 2.0

layer_num = [32, 32, 32, 32, 32, 32, 32, 32]
activations = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
network = FCNN_MSE(4, layer_num, 1, activations, weight_init_std = 1.0)

output_imgs = []
latent_values = np.arange(0.0, 1.0 + 0.01, 0.003)

print('latent value num : ' + str(len(latent_values)))
fig = plt.figure('Pattern Produced', figsize = (8, 8))

for i, cur_latent_value in enumerate(latent_values):
	output_img = generate_img_grayscale(network, cur_latent_value)
	print('latent value num ' + str(i) + ' generated')
	output_img_imshow = plt.imshow(output_img, animated = True, cmap = plt.get_cmap('gray'))
	output_imgs.append([output_img_imshow])

ani = animation.ArtistAnimation(fig, output_imgs, interval = 10, blit = True)
ani.save('PatternNetwork.mp4', writer = 'ffmpeg', fps = 30)