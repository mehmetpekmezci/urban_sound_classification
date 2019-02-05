import tensorflow as tf
import numpy as np

def load_dataset(data):
	'''
	Given data (as a dict), load the data into training data, testing data.
	Also output the time indices.
	'''
	np.random.seed(1) #maintain consistent results

	X = data['X']
	Y = data['FFT_X']
	t = data['t']

	indices = np.random.permutation(Y.shape[0])

	percent_training = .9
	num_training = int(percent_training * Y.shape[0])

	training_idx, test_idx = indices[:num_training], indices[num_training:]

	X_train = X[training_idx,:].T
	Y_train = Y[training_idx,:].T

	X_test = X[test_idx, :].T
	Y_test = Y[test_idx, :].T

	return (X_train, Y_train, X_test, Y_test, t, percent_training)