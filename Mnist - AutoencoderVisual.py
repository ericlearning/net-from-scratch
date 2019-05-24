import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from HypeNet.Networks.FCNN_MSE import FCNN_MSE
from HypeNet.Core.loadData import loadMnist
from HypeNet.Core.utils import *
import random

X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label = loadMnist()

image_num = 100
images = []
for i in range(image_num):
	images.append(X_val.T[random.randint(0, 10000 - 1)].reshape(784, 1))

#load the decoder network
decoder_DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/MnistAutoencoder/Decoder/'
decoder_network = FCNN_MSE(100, [500, 1000], 784, [ 'Relu', 'Relu', 'Sigmoid'], weight_init_std = 'he')
networkLoader(decoder_network, decoder_DIR)

#load the encoder network
encoder_DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/MnistAutoencoder/Encoder/'
encoder_network = FCNN_MSE(784, [1000, 500], 100, [ 'Relu', 'Relu', 'Relu'], weight_init_std = 'he')
networkLoader(encoder_network, encoder_DIR)

#set the figure
fig = plt.figure()

latents = []
for i in range(image_num):
	latents.append(encoder_network.predict(images[i]))

times = np.arange(0, 1, 0.1)

reconstructed_images = []
for i in range(image_num - 1):
	for cur_t in times:
		latent = lerp(latents[i], latents[i+1], cur_t)
		reconstruction = decoder_network.predict(latent).reshape(28, 28)
		im = plt.imshow(reconstruction, animated = True)
		reconstructed_images.append([im])

ani = animation.ArtistAnimation(fig, reconstructed_images, interval = 50, blit = True)
plt.show()