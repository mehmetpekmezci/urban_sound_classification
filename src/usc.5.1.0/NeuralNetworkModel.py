#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class NeuralNetworkModel :
 def __init__(self, session, logger, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE , keep_prob=DROP_OUT
             , learning_rate=LEARNING_RATE, mini_batch_size=MINI_BATCH_SIZE, number_of_time_slices=NUMBER_OF_TIME_SLICES,
              number_of_lstm_layers=NUMBER_OF_LSTM_LAYERS, lstm_state_size=LSTM_STATE_SIZE,lstm_forget_bias=LSTM_FORGET_BIAS):
   self.session               = session
   self.logger                = logger
   self.input_size            = input_size
   self.output_size           = output_size
   self.learning_rate         = learning_rate 
   self.mini_batch_size       = mini_batch_size
   self.number_of_time_slices = number_of_time_slices
   self.number_of_lstm_layers = number_of_lstm_layers
   self.lstm_state_size       = lstm_state_size
   self.lstm_forget_bias      = lstm_forget_bias
   self.keep_prob             = keep_prob
   self.keep_prob_constant    = keep_prob
   
   

   ##
   ## DEFINE PLACE HOLDER FOR REAL OUTPUT VALUES FOR TRAINING
   ##
   self.real_y_values=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values")
   self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")

   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER
   ##
   #self.x_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   self.x_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size,  self.number_of_time_slices, int(self.input_size/self.number_of_time_slices)), name="input")
   with tf.name_scope('input_reshape'):
    print("self.x_input.shape="+str(self.x_input.shape))
    #self.x_input_reshaped = tf.reshape(self.x_input, [self.mini_batch_size, self.number_of_time_slices, int(self.input_size/self.number_of_time_slices)])
    # Unstack to get a list of 'number_of_time_slices' tensors of shape (batch_size, input_size/number_of_time_slices,number_of_input_channels)
    #self.x_input_list = tf.unstack(self.x_input_reshaped, self.number_of_time_slices, 1)
    self.x_input_list = tf.unstack(self.x_input, self.number_of_time_slices, 1)
    print("self.x_input="+str(self.x_input_list))
    print("len(self.x_input)="+str(len(self.x_input_list)))
    print("self.x_input[0].shape"+str(self.x_input_list[0].shape))
    
   ##
   ## LSTM LAYERS
   ##
   with tf.name_scope("lstm"):
    ####lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_state_size)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_state_size, forget_bias=self.lstm_forget_bias)
    # create a RNN cell composed sequentially of a number of RNNCells
    ####multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.number_of_lstm_layers)
    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a tf.contrib.rnn.LSTMStateTuple for each cell : 
    #    tf.nn.rnn_cell.LSTMStateTuple(lstm_cell_state , lstm_hidden_state) 
    lstm_cell_with_dropout=tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
    #lstm_cell_with_dropout=tf.nn.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob_)
    lstm_outputs, lstm_state = tf.nn.static_rnn(lstm_cell_with_dropout, inputs=self.x_input_list, dtype=tf.float32)
    self.logger.info("lstm_outputs="+str( lstm_outputs))
    self.logger.info("lstm_state="+str( lstm_state))
    
    
    # Linear activation, using rnn inner loop last output
    
    weights = {'out': tf.Variable(tf.random_normal([self.lstm_state_size, self.output_size ]))}
    biases = {'out': tf.Variable(tf.random_normal([self.output_size ]))}
    self.y_outputs=tf.matmul(lstm_outputs[-1], weights['out']) + biases['out']
    # get last element of lstm_outputs = lstm_outputs[-1]= output for the last time step = final output = generated (guess) value  .
   
    ## HERE NETWORK DEFINITION IS FINISHED
    
    ###  NOW CALCULATE PREDICTED VALUE
    #with tf.name_scope('calculate_predictions'):
    # output_shape = tf.shape(self.y_outputs)
    # self.predictions = tf.nn.softmax(tf.reshape(self.y_outputs, [-1, self.output_size]))
     
   ##
   ## CALCULATE LOSS
   ##
    with tf.name_scope('calculate_loss'):
     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values,logits=self.y_outputs)
     self.loss = tf.reduce_mean(cross_entropy)

   ##
   ## SET OPTIMIZER
   ##
   with tf.name_scope('optimizer'):
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.99).minimize(self.loss)

   ##
   ## CALCULATE ACCURACY
   ##
   with tf.name_scope('calculate_accuracy'):
    correct_prediction = tf.equal(tf.argmax(self.y_outputs, 1),tf.argmax(self.real_y_values, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    self.accuracy = tf.reduce_mean(correct_prediction)


   ##
   ## SAVE NETWORK GRAPH TO A DIRECTORY
   ##
   with tf.name_scope('save_graph'):
    self.logger.info('Saving graph to: %s' % LOG_DIR_FOR_TF_SUMMARY)
    graph_writer = tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY)
    graph_writer.add_graph(tf.get_default_graph())
   
   

   ##
   ## INITIALIZE SESSION
   ##
   #checkpoint= tf.train.get_checkpoint_state(os.path.dirname(SAVE_DIR+'/usc_model'))
   #if checkpoint and checkpoint.model_checkpoint_path:
  #  saver.restore(self.session,checkpoint.model_checkpoint_path)
  # else : 
  #  self.session.run(tf.global_variables_initializer())

 def prepareData(self,data,augment):
  x_data=data[:,:4*SOUND_RECORD_SAMPLING_RATE]
  if augment==True :
    x_data=augment_random(x_data)
  ## generate gammatone filterbank from x_data
  x_data_reshaped = tf.reshape(x_data, [self.mini_batch_size, self.number_of_time_slices, int(self.input_size/self.number_of_time_slices)])
  for miniBatch in range(self.mini_batch_size):
   for timeSlice in range(self.number_of_time_slices):
       gammatone_specgram=get_gammatone_specgram(x_data_reshaped[miniBatch,timeSlice])
       print(gammatone_specgram);
       x_data_gammatone[miniBatch,timeSlice]=get_gammatone_specgram(x_data_reshaped[miniBatch,timeSlice])    

  y_data=data[:,4*SOUND_RECORD_SAMPLING_RATE]
  y_data_one_hot_encoded=one_hot_encode_array(y_data)
  return x_datai_gammatone,y_data_one_hot_encoded

 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  self.optimizer.run(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob:self.keep_prob_constant})
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingAccuracy = self.accuracy.eval(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob: 1.0})
  return trainingTime,trainingAccuracy,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data,y_data=self.prepareData(data,augment) 
  testAccuracy = self.accuracy.eval(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob: 1.0})
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  



