
import ann
import numpy as np
from itertools import islice
from random import random
from cv2 import imshow,waitKey,resize,INTER_NEAREST
import sys

def randn(lst, size):
	ln = len(lst)
	for i in range(size):
		idx = int(random() * (ln - i)) + i
		tmp = lst[idx]
		yield tmp
		lst[idx] = lst[i]
		lst[i] = tmp

class SignalParameters:
	def __init__(self, look_back, extrap, hidden_layers,\
			training_iterations, learning_rate=1., stochastic=True,\
			batch_size=0):
		self.look_back = look_back
		self.extrap = extrap
		self.hidden_layers = hidden_layers
		self.iterations = training_iterations
		self.learning_rate = learning_rate
		self.stochastic = stochastic
		self.batch_size = batch_size

EXAMPLE_PARAMETERS = {\
	'music': SignalParameters(44100, 44100//128, [256, 256], 2,\
								batch_size=1024),\
	'stocks': SignalParameters(128, 4, [64, 64, 64], 1000),\
}

class SignalExtrapolator:
	def __init__(self, parameters):
		self.params = parameters
		p = self.params
		self.net = ann.ANN([p.look_back]+p.hidden_layers+[p.extrap],\
			activation='tanh+ax', learning_rate=p.learning_rate,\
			rand_scale=.1)
	def get_look_back(self):
		return self.params.look_back
	def train(self, signals):
		samples = []
		lb,ex = self.params.look_back,self.params.extrap
		tot_size = lb+ex
		for sig in signals:
			for i in range(len(sig)-tot_size):
				samples.append((i,sig))
		end = self.params.batch_size
		if not self.params.stochastic or end <= 0:
			end = len(samples)
		lr = 1./end
		for i in range(self.params.iterations):
			print("Training iteration %d"%(i+1), file=sys.stderr)
			if self.params.stochastic:
				itr = randn(samples, end)
			else:
				itr = islice(samples, end)
			for i,sig in itr:
				inp,res = np.array(sig[i:i+lb],\
									dtype=np.float32),\
								np.array(sig[i+lb:i+tot_size],\
									dtype=np.float32)
				self.net.train(inp, res, lr)
			for i in range(self.net.get_layer_count()):
				imshow('Layer %d'%i,\
					resize(self.net.get_layer_image(i), (300,300),\
					interpolation=INTER_NEAREST))
			waitKey(50)
	def extrapolate(self, sig):
		return self.net.forward(sig)
