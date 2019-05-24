import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

DIR = os.path.dirname(os.path.abspath(__file__)) + '/SavedNetwork/FashionMnistCNN/'
filters = np.load(DIR + 'learnable_W2.npy')
N, C, H, W = filters.shape

h, w = 5, 5
fig = plt.figure()

cnt = 1
isBreak = False

for i in range(N):
	for j in range(C):
		fig.add_subplot(h, w, cnt)
		plt.imshow(filters[i, j])
		print(cnt)
		cnt+=1
		if(h*w<cnt):
			isBreak = True
			break

	if(isBreak == True):
		break

plt.show()