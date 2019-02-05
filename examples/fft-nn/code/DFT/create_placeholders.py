import tensorflow as tf
import numpy as np

def create_placeholders(n_x, n_y):

	"""
    Creates the placeholders for the tensorflow session.

    """


	X = tf.placeholder(tf.float32, [n_x, None])
	Y = tf.placeholder(tf.float32, [n_y, None])
	keep_prob = tf.placeholder(tf.float32) #used in dropout

	return X, Y, keep_prob