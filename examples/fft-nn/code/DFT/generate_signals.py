import tensorflow as tf
import numpy as np
import scipy as sc

def generate_signals(num_X = 10, max_bandwidth = 10, max_num_sinusoids = 100):
	'''
	Given the number of examples, the maximum bandwidth requested, and the maximum number of sinusoids in one example,
	we return a dictionary of randomly generated sinusoids (subject to the constraints described above) X, and their
	associated bandwidths Y. We also return the time indices t.
	'''

	np.random.seed(1)
	tt = np.linspace(0, 1, 100)
	tt_length = tt.shape[0]

	X = []
	Y = []

	for i in range(num_X):

		num_sinusoids = np.random.randint(1, max_num_sinusoids)

		freqs = []
		signal = 0
		for j in range(num_sinusoids):
			rand_freq = max_bandwidth * np.random.rand()
			freqs.append(rand_freq)

			rand_offset = 10*np.random.randn()

			sine_or_cosine = np.random.randint(2) #1/2 chance for sine, 1/2 chance for cosine

			noise_or_nonoise = np.random.randint(20) #1/20 of the signals have noise

			if sine_or_cosine == 1:
				signal += np.sin(2 * np.pi * rand_freq * tt)
			else:
				signal += np.cos(2 * np.pi * rand_freq * tt)

			if noise_or_nonoise == 1:
				signal += np.random.randn(tt_length)

		X = np.hstack((X,signal))

	X = X.reshape(num_X, tt_length)

	data = dict()
	data['t'] = tt 
	data['X'] = X
	data['FFT_X'] = np.real(sc.linalg.dft(tt_length).dot(X.T).T)
	return data
