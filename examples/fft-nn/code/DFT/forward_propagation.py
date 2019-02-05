import tensorflow as tf
import numpy as np

def forward_propagation(X, keep_prob, parameters):
	"""
	Implements the forward propagation for the model.

	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "W6", "b6". 
	The shapes are given in initialize_parameters.

	Returns:
	Z_L -- the output of the last unit
	"""

	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	W4 = parameters['W4']
	b4 = parameters['b4']
	W5 = parameters['W5']
	b5 = parameters['b5']
	W6 = parameters['W6']
	b6 = parameters['b6']

	Z1 = tf.add(tf.matmul(W1, X),b1)
	A1 = tf.nn.dropout(Z1, keep_prob)

	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.dropout(Z2, keep_prob)

	Z3 = tf.add(tf.matmul(W3,A2), b3)
	# A3 = tf.nn.dropout(Z3, keep_prob)
	
	# Z4 = tf.add(tf.matmul(W4,A3), b4) 
	# A4 = tf.nn.dropout(Z4, keep_prob)
	
	# Z5 = tf.add(tf.matmul(W5,A4), b5) 
	# A5 = tf.nn.dropout(Z5, keep_prob)

	# '''...last layer'''        
	# Z_L = tf.add(tf.matmul(W6,A5), b6)
	return Z3


	