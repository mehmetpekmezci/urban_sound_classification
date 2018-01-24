#!/usr/bin/env python3
from header import *
from data import *


class ModelNetwork:
	def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.0003, name="rnn"):
		self.scope = name
		self.in_size = in_size
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.out_size = out_size
		self.session = session
		self.learning_rate = tf.constant( learning_rate )
		# Last state of LSTM, used when running the network in TEST mode
		self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))
		with tf.variable_scope(self.scope):
			## (batch_size, timesteps, in_size)
			self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
			self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")
			# LSTM
			self.lstm_cells = [ tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
			self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)
			# Iteratively compute output of recurrent network
			outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)
			# Linear activation (FC layer on top of the LSTM net)
			self.rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=0.01 ))
			self.rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=0.01 ))
			outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
			network_output = ( tf.matmul( outputs_reshaped, self.rnn_out_W ) + self.rnn_out_B )
			batch_time_shape = tf.shape(outputs)
			#self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )
			self.final_outputs = tf.reshape( network_output, (batch_time_shape[0], batch_time_shape[1], self.out_size) )
			## Training: provide target outputs for supervised training.
			self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
			y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])
			self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long) )
			#self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)
			self.train_op  = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
			self.accuracy  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_output,1),tf.argmax(y_batch_long,1)), tf.float32))
	## Input: X is a single element, not a list!
	def run_step(self, x, init_zero_state=True):
		## Reset the initial state of the network.
		if init_zero_state:
			init_value = np.zeros((self.num_layers*2*self.lstm_size,))
		else:
			init_value = self.lstm_last_state
		out, next_lstm_state,accuracy = self.session.run([self.final_outputs, self.lstm_new_state,self.accuracy], feed_dict={self.xinput:[x], self.lstm_init_value:[init_value]   } )
		#self.lstm_last_state = next_lstm_state[0]
		return out[0][0],accuracy
	## xbatch must be (batch_size, timesteps, input_size)
	## ybatch must be (batch_size, timesteps, output_size)
	def train_batch(self, xbatch, ybatch):
		init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))
		accuracy,_ = self.session.run([self.accuracy, self.train_op], feed_dict={self.xinput:xbatch, self.y_batch:ybatch, self.lstm_init_value:init_value   } )
		return accuracy
