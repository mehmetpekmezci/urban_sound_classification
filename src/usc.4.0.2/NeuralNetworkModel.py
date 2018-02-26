#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class NeuralNetworkModel :
 def __init__(self, session, logger, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE , cnn_kernel_counts=CNN_KERNEL_COUNTS, cnn_kernel_x_sizes=CNN_KERNEL_X_SIZES, 
              cnn_kernel_y_sizes=CNN_KERNEL_Y_SIZES,cnn_stride_x_sizes=CNN_STRIDE_X_SIZES,cnn_stride_y_sizes=CNN_STRIDE_Y_SIZES,cnn_pool_x_sizes=CNN_POOL_X_SIZES,cnn_pool_y_sizes=CNN_POOL_Y_SIZES, 
              learning_rate=LEARNING_RATE, mini_batch_size=MINI_BATCH_SIZE, number_of_time_slices=NUMBER_OF_TIME_SLICES,number_of_lstm_layers=NUMBER_OF_LSTM_LAYERS, lstm_state_size=LSTM_STATE_SIZE
              ,number_of_fully_connected_layer_neurons=NUMBER_OF_FULLY_CONNECTED_NEURONS):

   ##
   ## SET CLASS ATTRIBUTES WITH THE GIVEN INPUTS
   ##
   self.session               = session
   self.logger                = logger
   self.input_size            = input_size
   self.output_size           = output_size
   self.cnn_kernel_counts     = cnn_kernel_counts
   self.cnn_kernel_x_sizes    = cnn_kernel_x_sizes
   self.cnn_kernel_y_sizes    = cnn_kernel_y_sizes
   self.cnn_stride_x_sizes    = cnn_stride_x_sizes
   self.cnn_stride_y_sizes    = cnn_stride_y_sizes
   self.cnn_pool_x_sizes      = cnn_pool_x_sizes
   self.cnn_pool_y_sizes      = cnn_pool_y_sizes
   self.learning_rate         = learning_rate 
   self.mini_batch_size       = mini_batch_size
   self.number_of_time_slices = number_of_time_slices
   self.number_of_lstm_layers = number_of_lstm_layers
   self.lstm_state_size       = lstm_state_size
   self.number_of_fully_connected_layer_neurons=number_of_fully_connected_layer_neurons


   
   ##
   ## INITIALIZE SESSION
   ##
   self.session.run(tf.global_variables_initializer())

   ##
   ## DEFINE PLACE HOLDER FOR REAL OUTPUT VALUES FOR TRAINING
   ##
   self.real_y_values=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values")

   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER
   ##
   number_of_input_channels=1
   self.x_input                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   with tf.name_scope('input_reshape'):
    print("self.x_input.shape="+str(self.x_input.shape))
    self.x_input = tf.reshape(self.x_input, [self.mini_batch_size, self.number_of_time_slices, int(self.input_size/self.number_of_time_slices), number_of_input_channels])
    ## image analogy : [ batch_size, image_height, image_width, RGB_channel_3 ]    
    print("self.x_input.shape="+str(self.x_input.shape))
    
   ##
   ## CNN LAYERS
   ##
   previous_level_convolution_output = self.x_input

   for cnnLayerNo in range(len(self.cnn_kernel_counts)) :
     cnnLayerName    = "cnn-"+str(cnnLayerNo)     
     cnnKernelCount  = self.cnn_kernel_counts[cnnLayerNo]   # cnnKernelCount tane cnnKernelSizeX * cnnKernelSizeY lik convolution kernel uygulanacak , sonucta 64x1x88200 luk tensor cikacak.
     cnnKernelSizeX  = self.cnn_kernel_x_sizes[cnnLayerNo]
     cnnKernelSizeY  = self.cnn_kernel_y_sizes[cnnLayerNo]         
     cnnStrideSizeX  = self.cnn_stride_x_sizes[cnnLayerNo] 
     cnnStrideSizeY  = self.cnn_stride_y_sizes[cnnLayerNo]                     
     cnnPoolSizeX    = self.cnn_pool_x_sizes[cnnLayerNo]          
     cnnPoolSizeY    = self.cnn_pool_y_sizes[cnnLayerNo]      
     cnnOutputChannel= cnnKernelCount   
     if cnnLayerNo == 0 :
       cnnInputChannel = 1
     else :
       cnnInputChannel = self.cnn_kernel_counts[int(cnnLayerNo-1)]   
     with tf.name_scope(cnnLayerName+"-convolution"):
       ## WEIGHT    
       W = tf.Variable(tf.truncated_normal([cnnKernelSizeX, cnnKernelSizeY, cnnInputChannel, cnnOutputChannel], stddev=0.1))
       print(W)
       ## BIAS
       B = tf.Variable(tf.constant(0.1, shape=[cnnOutputChannel]))
       #Based on conv2d doc:
       #    shape of input = [batch, in_height, in_width, in_channels]
       #    shape of filter = [filter_height, filter_width, in_channels, out_channels]
       #    Last dimension of input and third dimension of filter represents the number of input channels
       C = tf.nn.conv2d(previous_level_convolution_output,W,strides=[cnnStrideSizeX, cnnStrideSizeY, 1, 1], padding='SAME')+B
       self.logger.info(cnnLayerName+"_C.shape="+str(C.shape))
     print(C)
     with tf.name_scope(cnnLayerName+"-relu"):  
       H = tf.nn.relu(C)
       self.logger.info(cnnLayerName+"_H.shape="+str(H.shape))

     with tf.name_scope(cnnLayerName+"-pool"):
       # Max Pooling layer - downsamples by pool_length.
       # pooled by 1 x cnnPoolSize
       P = tf.nn.max_pool(H, ksize=[1, cnnPoolSizeX,cnnPoolSizeY, 1],strides=[1, cnnPoolSizeX,cnnPoolSizeY, 1], padding='SAME') 
       self.logger.info(cnnLayerName+".H_pooled.shape="+str(P.shape))
       ## put the output of this layer to the next layer's input layer.
       previous_level_convolution_output=P
       print(P)

   cnn_last_layer_output=previous_level_convolution_output


   with tf.name_scope("cnn-result-reshape"):
    ## flattened the result :  batch_size and time_slices stays as they are, combined "filtered value" and kernels in the last column
    cnnFlatResult = tf.reshape(cnn_last_layer_output, [self.mini_batch_size, self.number_of_time_slices, int(cnn_last_layer_output.shape[2]*cnn_last_layer_output.shape[3])])
    #previous_level_conv_output_flat=tf.squeeze(previous_level_conv_output)
    self.logger.info("cnnFlatResult.shape="+str( cnnFlatResult.shape))


   ##
   ## LSTM LAYERS
   ##
   with tf.name_scope("lstm"):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_state_size)
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cells] * self.number_of_lstm_layers)
    ## unstack the cnnFlatResult as time slices to feed into lstm.
    cnnFlatResultUnstacked = tf.unstack(cnnFlatResult, self.number_of_time_slices, 1)   
    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a tf.contrib.rnn.LSTMStateTuple for each cell : tf.nn.rnn_cell.LSTMStateTuple(lstm_cell_state , lstm_hidden_state) 
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=cnnFlatResultUnstacked, dtype=tf.float32)
    self.logger.info("lstm_outputs="+str( lstm_outputs))
    self.logger.info("lstm_state="+str( lstm_state))

   ##
   ## FULLY CONNECTED LAYERS
   ##Linear activation (FC layer on top of the LSTM net)
   # lstm output is reshaped to be able to feed the fc net.
   with tf.name_scope('lstm_to_fc_reshape'):
    lstm_outputs_reshaped = tf.reshape( lstm_outputs, [-1, self.lstm_state_size] )
    self.logger.info("lstm_outputs_reshaped="+str( lstm_outputs_reshaped))


   with tf.name_scope('fc1'):
    W_fc1 =  tf.Variable( tf.truncated_normal([int(lstm_outputs_reshaped.shape[1]), self.number_of_fully_connected_layer_neurons], stddev=0.1))
    self.logger.info("W_fc1.shape="+str(W_fc1.shape))
    B_fc1 = tf.Variable(tf.constant(0.1, shape=[self.number_of_fully_connected_layer_neurons]))
    self.logger.info("B_fc1.shape="+str(B_fc1.shape))
    matmul_fc1=tf.matmul(lstm_outputs_reshaped, W_fc1)+b_fc1
    self.logger.info("matmul_fc1.shape="+str(matmul_fc1.shape))
    
   with tf.name_scope('fc1_batch_normlalization'):    
    batch_mean, batch_var = tf.nn.moments(matmul_fc1,[0])
    scale = tf.Variable(tf.ones(self.number_of_fully_connected_layer_neurons))
    beta = tf.Variable(tf.zeros(self.number_of_fully_connected_layer_neurons))
    batch_normalization_fc1 = tf.nn.batch_normalization(matmul_fc1,batch_mean,batch_var,beta,scale,epsilon)
    self.logger.info("batch_normalization_fc1.shape="+str(batch_normalization_fc1.shape))

   with tf.name_scope('fc1_batch_normalized_relu'):    
    h_fc1 = tf.nn.relu( batch_normalization_fc1 )
    self.logger.info("h_fc1.shape="+str(fc1_output.shape))

   # Dropout - controls the complexity of the model, prevents co-adaptation of features.
   with tf.name_scope('fc1_dropout'):    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    self.logger.info("h_fc1_drop.shape="+str(h_fc1_drop.shape))

    # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS(1024) features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('fc2'):
    W_fc2 =  tf.Variable( tf.truncated_normal([int(lstm_outputs_reshaped.shape[1]), self.number_of_fully_connected_layer_neurons], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self._output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs =tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    self.logger.info("self.y_outputs.shape="+str(self.y_outputs.shape))
    
   
    ## HERE NETWORK DEFINITION IS FINISHED
    
    ##  NOW CALCULATE PREDICTED VALUE
    with tf.name_scope('calculate_predictions'):
     output_shape = tf.shape(self.y_outputs)
     self.predictions = tf.nn.softmax(tf.reshape(self.y_outputs, [-1, self.output_size]))
     
   ##
   ## CALCULATE LOSS
   ##
    with tf.name_scope('calculate_loss'):
     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values,logits=slef.y_outputs)
     self.loss = tf.reduce_mean(cross_entropy)

   ##
   ## SET OPTIMIZER
   ##
   with tf.name_scope('optimizer'):
    self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

   ##
   ## CALCULATE ACCURACY
   ##
   with tf.name_scope('calculate_accuracy'):
    correct_prediction = tf.equal(tf.argmax(self.real_y_values, 1), tf.argmax(self.y_outputs, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    self.accuracy = tf.reduce_mean(correct_prediction)


   ##
   ## SAVE NETWORK GRAPH TO A DIRECTORY
   ##
   with tf.name_scope('save_graph'):
    self.logger.info('Saving graph to: %s' % LOG_DIR_FOR_TF_SUMMARY)
    graph_writer = tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY)
    graph_writer.add_graph(tf.get_default_graph())



 def prepareData(self,data):
  x_data=augment_random(data[:,:4*SOUND_RECORD_SAMPLING_RATE])
  y_data=data[:,4*SOUND_RECORD_SAMPLING_RATE]
  y_data_one_hot_encoded=one_hot_encode_array(y_data)
  return x_data,y_data

 def train(self,data):
  trainingTimeStart = int(round(time.time())) 
  x_data,y_data=prepareData(data)
  self.optimizer.run(feed_dict={x: x_data, y: y_data, keep_prob: DROP_OUT})
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingAccuracy = accuracy.eval(feed_dict={x: x_data, y:y_data_one_hot_encoded, keep_prob: 1.0})
  return trainingTime,trainingAccuracy
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  x_data,y_data=prepareData(data) 
  testAccuracy = accuracy.eval(feed_dict={x: test_x_data, y:test_y_data_one_hot_encoded, keep_prob: 1.0})
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  



