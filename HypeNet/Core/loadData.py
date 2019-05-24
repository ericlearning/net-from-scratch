import os
import cv2
import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import *
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
from skimage.draw import *
from HypeNet.Core.utils import *
import pandas as pd

#Mnist Loader
def loadMnist(flatten = True):
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	train_DIR = DIR + '/Data/MNIST/mnist_train.csv'
	val_DIR = DIR + '/Data/MNIST/mnist_test.csv'

	train_dataset = pd.read_csv(train_DIR, header = None)
	val_dataset = pd.read_csv(val_DIR, header = None)

	train_dataset = train_dataset.values
	val_dataset = val_dataset.values

	Y_train_label = train_dataset[:, :1].T 			#(1, 60000)
	Y_val_label = val_dataset[:, :1].T 				#(1, 10000)

	X_train = train_dataset[:, 1:].T / 255.0
	Y_train = categorical(Y_train_label, 10) 		#(10, 60000)
	X_val = val_dataset[:, 1:].T / 255.0
	Y_val = categorical(Y_val_label, 10) 			#(10, 10000)

	if(flatten == True):
		return X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label
	else:
		X_train = X_train.T
		X_val = X_val.T
		return X_train.reshape(60000, 1, 28, 28), Y_train, X_val.reshape(10000, 1, 28, 28), Y_val, Y_train_label, Y_val_label

#Fashion Mnist Loader
def loadFashionMnist(flatten = True):
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	train_DIR = DIR + '/Data/FashionMNIST/fashion-mnist_train.csv'
	val_DIR = DIR + '/Data/FashionMNIST/fashion-mnist_test.csv'

	train_dataset = pd.read_csv(train_DIR, header = None, low_memory = False)
	val_dataset = pd.read_csv(val_DIR, header = None, low_memory = False)

	train_dataset = train_dataset.values[1:].T
	val_dataset = val_dataset.values[1:].T

	X_train = train_dataset[1:]
	X_val = val_dataset[1:]
	Y_train_label = train_dataset[0]
	Y_train_label = Y_train_label.reshape(1, Y_train_label.shape[0])
	Y_val_label = val_dataset[0]
	Y_val_label = Y_val_label.reshape(1, Y_val_label.shape[0])

	X_train = X_train.astype('float32') / 255.0
	X_val = X_val.astype('float32') / 255.0
	Y_train_label = Y_train_label.astype('int')
	Y_val_label = Y_val_label.astype('int')
	
	Y_train = categorical(Y_train_label, 10)
	Y_val = categorical(Y_val_label, 10)

	if(flatten == True):
		return X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label
	else:
		X_train = X_train.T
		X_val = X_val.T
		return X_train.reshape(60000, 1, 28, 28), Y_train, X_val.reshape(10000, 1, 28, 28), Y_val, Y_train_label, Y_val_label


#CatDog Preprocesser & Loader
def preprocess_CatDog():
	ROWS = 256
	COLS = 256
	CHANNELS = 3
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	train_DIR = DIR + '/Data/CatDog/train/'
	test_DIR = DIR + '/Data/CatDog/test/'
	save_DIR = DIR + '/Data/CatDog/preprocessed/'

	train_images = [train_DIR + i for i in os.listdir(train_DIR)]
	test_images = [test_DIR + i for i in os.listdir(test_DIR)]

	X_train = []
	Y_train = []
	for image_path in train_images:
		cur_image = cv2.imread(image_path)
		cur_image = cv2.resize(cur_image, (ROWS, COLS), interpolation = cv2.INTER_CUBIC)
		X_train.append(cur_image)
		if('dog' in image_path):
			Y_train.append(1)
		elif('cat' in image_path):
			Y_train.append(0)

	X_test = []
	for image_path in test_images:
		cur_image = cv2.imread(image_path)
		cur_image = cv2.resize(cur_image, (ROWS, COLS), interpolation = cv2.INTER_CUBIC)
		X_test.append(cur_image)


	X_train = np.asarray(X_train)
	X_train = X_train.transpose(1, 2, 3, 0)
	X_train = X_train.reshape(ROWS * COLS * CHANNELS, X_train.shape[3])
	Y_train = np.asarray(Y_train)
	Y_train = categorical(Y_train.reshape(1, Y_train.shape[0]), 2)

	X_test = np.asarray(X_test)
	X_test = X_test.transpose(1, 2, 3, 0)
	X_test = X_test.reshape(ROWS * COLS * CHANNELS, X_test.shape[3])

	X_train, Y_train = shuffle_data(X_train, Y_train)

	X_train, X_val, Y_train, Y_val = train_test_split(X_train.T, Y_train.T, test_size = 0.2)
	X_train = X_train.T / 255.0
	Y_train = Y_train.T
	X_val = X_val.T / 255.0
	Y_val = Y_val.T
	X_test = X_test / 255.0

	np.save(save_DIR + 'X_train_256.npy', X_train)
	np.save(save_DIR + 'Y_train_256.npy', Y_train)
	np.save(save_DIR + 'X_val_256.npy', X_val)
	np.save(save_DIR + 'Y_val_256.npy', Y_val)
	np.save(save_DIR + 'X_test_256.npy', X_test)
	
def loadCatDog(flatten = True):
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	data_DIR = DIR + '/Data/CatDog/preprocessed/'

	X_train_DIR = data_DIR + 'X_train.npy'
	Y_train_DIR = data_DIR + 'Y_train.npy'
	X_val_DIR = data_DIR + 'X_val.npy'
	Y_val_DIR = data_DIR + 'Y_val.npy'
	X_test_DIR = data_DIR + 'X_test.npy'

	X_train = np.load(X_train_DIR)
	Y_train = np.load(Y_train_DIR)
	X_val = np.load(X_val_DIR)
	Y_val = np.load(Y_val_DIR)
	X_test = np.load(X_test_DIR)

	if(flatten == True):
		return X_train, Y_train, X_val, Y_val, X_test
	else:
		N_train = X_train.shape[1]
		N_val = X_val.shape[1]
		N_test = X_test.shape[1]
		X_train = X_train.T
		X_val = X_val.T
		X_test = X_test.T
		return X_train.reshape(N_train, 3, 64, 64), Y_train, X_val.reshape(N_val, 3, 64, 64), Y_val, X_test.reshape(N_test, 3, 64, 64)

def loadCatDog_256(flatten = True):
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	data_DIR = DIR + '/Data/CatDog/preprocessed_256/'

	X_train_DIR = data_DIR + '256_X_train.npy'
	Y_train_DIR = data_DIR + '256_Y_train.npy'
	X_val_DIR = data_DIR + '256_X_val.npy'
	Y_val_DIR = data_DIR + '256_Y_val.npy'
	X_test_DIR = data_DIR + '256_X_test.npy'

	X_train = np.load(X_train_DIR)
	Y_train = np.load(Y_train_DIR)
	X_val = np.load(X_val_DIR)
	Y_val = np.load(Y_val_DIR)
	X_test = np.load(X_test_DIR)

	if(flatten == True):
		return X_train, Y_train, X_val, Y_val, X_test
	else:
		N_train = X_train.shape[1]
		N_val = X_val.shape[1]
		N_test = X_test.shape[1]
		X_train = X_train.T
		X_val = X_val.T
		X_test = X_test.T
		return X_train.reshape(N_train, 3, 256, 256), Y_train, X_val.reshape(N_val, 3, 256, 256), Y_val, X_test.reshape(N_test, 3, 256, 256)

#SignLanguage Loader
def loadSign():
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	X_DIR = DIR + '/Data/Sign Language/X.npy'
	Y_DIR = DIR + '/Data/Sign Language/Y.npy'

	X = np.load(X_DIR).reshape((2062, 64 * 64))
	Y = np.load(Y_DIR)

	X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

	X_train = X_train.T
	X_val = X_val.T
	Y_train = Y_train.T
	Y_val = Y_val.T

	Y_train_label = np.argmax(Y_train, axis = 0)
	Y_val_label = np.argmax(Y_val, axis = 0)

	return X_train, Y_train, X_val, Y_val, Y_train_label, Y_val_label

#Doodle Generator & Loader
def drawEllipse(img):
	r = random.randint(7, 28 - 1)
	c = random.randint(7, 28 - 1)
	r_rad = random.randint(0, 15)
	c_rad = random.randint(0, 15)
	rr, cc = ellipse_perimeter(r, c, r_rad, c_rad, orientation = random.uniform(0, math.pi * 2), shape = img.shape)
	img[rr, cc] = 1
	return img

def drawBezier(img):
	r0, c0 = random.randint(0, 28 - 1), random.randint(0, 28 - 1)
	r1, c1 = random.randint(0, 28 - 1), random.randint(0, 28 - 1)
	r2, c2 = random.randint(0, 28 - 1), random.randint(0, 28 - 1)
	weight = random.uniform(0, 10)
	rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, weight = weight, shape = img.shape)
	img[rr, cc] = 1
	return img

def drawLine(img):
	r0, c0 = random.randint(0, 28 - 1), random.randint(0, 28 - 1)
	r1, c1 = random.randint(0, 28 - 1), random.randint(0, 28 - 1)
	rr, cc = line(r0, c0, r1, c1)
	img[rr, cc] = 1
	return img

def saveDoodles(doodle_num, doodle_val_num):
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	save_DIR = DIR + '/Data/Doodle/'

	doodles = []
	for i in range(doodle_num):
		img = np.zeros((28, 28))
		img = drawEllipse(img)
		img = drawBezier(img)
		img = drawLine(img)
		img = grey_dilation(img, (3, 3))
		img = gaussian(img, sigma = 0.5)
		img = grey_erosion(img, (3, 3))
		img = img.reshape(784)
		doodles.append(img)

	doodles_val = []
	for i in range(doodle_val_num):
		img = np.zeros((28, 28))
		img = drawEllipse(img)
		img = drawBezier(img)
		img = drawLine(img)
		img = grey_dilation(img, (3, 3))
		img = gaussian(img, sigma = 0.5)
		img = grey_erosion(img, (3, 3))
		img = img.reshape(784)
		doodles_val.append(img)

	doodles = np.asarray(doodles).T
	doodles_val = np.asarray(doodles_val).T
	np.save(save_DIR + 'Doodles.npy', doodles)
	np.save(save_DIR + 'Doodles_val.npy', doodles_val)

def loadDoodles():
	DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	data_DIR = DIR + '/Data/Doodle/'

	doodles_DIR = data_DIR + 'Doodles.npy'
	doodles_val_DIR = data_DIR + 'Doodles_val.npy'

	doodles = np.load(doodles_DIR)
	doodles_val = np.load(doodles_val_DIR)

	return doodles, doodles_val