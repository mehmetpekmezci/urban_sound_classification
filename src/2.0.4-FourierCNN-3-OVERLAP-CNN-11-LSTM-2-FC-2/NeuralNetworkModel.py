#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class NeuralNetworkModel :
 def __init__(self, session, logger, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE , cnn_kernel_counts=CNN_KERNEL_COUNTS, 
              cnn_kernel_timeslice_sizes=CNN_KERNEL_TIMESLICE_SIZES,cnn_kernel_x_sizes=CNN_KERNEL_X_SIZES,cnn_kernel_y_sizes=CNN_KERNEL_Y_SIZES,
              cnn_stride_timeslice_sizes=CNN_STRIDE_TIMESLICE_SIZES,cnn_stride_x_sizes=CNN_STRIDE_X_SIZES,cnn_stride_y_sizes=CNN_STRIDE_Y_SIZES,
              cnn_pool_timeslice_sizes=CNN_POOL_TIMESLICE_SIZES,cnn_pool_x_sizes=CNN_POOL_X_SIZES,cnn_pool_y_sizes=CNN_POOL_Y_SIZES, 
              learning_rate=LEARNING_RATE, mini_batch_size=MINI_BATCH_SIZE,  frequency_slice_length=FREQUENCY_SLICE_LENGTH,frequency_slice_overlap_length=FREQUENCY_SLICE_OVERLAP_LENGTH,
              number_of_lstm_layers=NUMBER_OF_LSTM_LAYERS, lstm_state_size=LSTM_STATE_SIZE,lstm_forget_bias=LSTM_FORGET_BIAS,
              epsilon=EPSILON,keep_prob_constant=KEEP_PROB,number_of_fully_connected_layer_neurons=NUMBER_OF_FULLY_CONNECTED_NEURONS,
              fourier_cnn_layers=FOURIER_CNN_LAYERS):

   ##
   ## SET CLASS ATTRIBUTES WITH THE GIVEN INPUTS
   ##
   self.session                       = session
   self.logger                        = logger
   self.input_size                    = input_size
   self.input_size_y                  = 1
   self.output_size                   = output_size
   self.cnn_kernel_counts             = cnn_kernel_counts
   self.cnn_kernel_timeslice_sizes    = cnn_kernel_timeslice_sizes
   self.cnn_kernel_x_sizes            = cnn_kernel_x_sizes
   self.cnn_kernel_y_sizes            = cnn_kernel_y_sizes
   self.cnn_stride_timeslice_sizes    = cnn_stride_timeslice_sizes
   self.cnn_stride_x_sizes            = cnn_stride_x_sizes
   self.cnn_stride_y_sizes            = cnn_stride_y_sizes
   self.cnn_pool_timeslice_sizes      = cnn_pool_timeslice_sizes
   self.cnn_pool_x_sizes              = cnn_pool_x_sizes
   self.cnn_pool_y_sizes              = cnn_pool_y_sizes
   self.learning_rate                 = learning_rate 
   self.mini_batch_size               = mini_batch_size
   self.frequency_slice_length        = frequency_slice_length
   self.frequency_slice_overlap_length= frequency_slice_overlap_length
   self.number_of_lstm_layers         = number_of_lstm_layers
   self.lstm_state_size               = lstm_state_size
   self.lstm_forget_bias              = lstm_forget_bias
   self.keep_prob_constant            = keep_prob_constant
   self.epsilon                       = epsilon
   self.number_of_fully_connected_layer_neurons=number_of_fully_connected_layer_neurons
   self.fourier_cnn_layers            =fourier_cnn_layers


   

   ##
   ## DEFINE PLACE HOLDER FOR REAL OUTPUT VALUES FOR TRAINING
   ##
   self.real_y_values=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values")

   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER, RESHAPE INPUT AND CUT IT INTO TIME SLICES
   ##
   number_of_input_channels=1
   self.x_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   last_layer_output=self.x_input

   ##
   ## RESHAPE
   ##
   with tf.name_scope('input_reshape'):
     self.x_input_reshaped = tf.reshape(last_layer_output, [self.mini_batch_size, self.input_size_y, last_layer_output.shape[1], number_of_input_channels])
     self.logger.info("self.x_input_reshaped.shape="+str(self.x_input_reshaped.shape))
     previous_level_convolution_output = self.x_input_reshaped

   ##
   ## FOURIER  LAYERS
   ##
   with tf.name_scope('fourier_CNN'):
    for fourierCNNLayerNo in range(len(self.fourier_cnn_layers)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "fourier-cnn-"+str(fourierCNNLayerNo)
     cnnKernelCount  = self.fourier_cnn_layers[fourierCNNLayerNo]
     # cnnKernelCount tane cnnKernelSizeX * cnnKernelSizeY lik convolution kernel uygulanacak , sonucta 64x1x88200 luk tensor cikacak.
     cnnKernelSizeX  = 1
     cnnKernelSizeY  = 3
     cnnStrideSizeX  = 1
     cnnStrideSizeY  = 2
     cnnPoolSizeX    = 1
     cnnPoolSizeY    = 2
     cnnOutputChannel= cnnKernelCount
     if fourierCNNLayerNo == 0 :
       cnnInputChannel = 1
     else :
       cnnInputChannel = self.fourier_cnn_layers[int(fourierCNNLayerNo-1)]


     with tf.name_scope(cnnLayerName+"-convolution"):
       W = tf.Variable(tf.truncated_normal([cnnKernelSizeX, cnnKernelSizeY, cnnInputChannel, cnnOutputChannel], stddev=0.1))
       B = tf.Variable(tf.constant(0.1, shape=[cnnOutputChannel]))
       C = tf.nn.conv2d(previous_level_convolution_output,W,strides=[1,cnnStrideSizeX, cnnStrideSizeY, 1], padding='SAME')+B

       self.logger.info(cnnLayerName+"_C.shape="+str(C.shape)+"  W.shape="+str(W.shape)+ "  cnnStrideSizeX="+str(cnnStrideSizeX)+" cnnStrideSizeY="+str(cnnStrideSizeY))

     ## no relu,  fourier transformation is linear.
     H=C

     #with tf.name_scope(cnnLayerName+"-relu"):  
     #  H = tf.nn.relu(C)
     #  self.logger.info(cnnLayerName+"_H.shape="+str(H.shape))

     if cnnPoolSizeY != 1 :
      with tf.name_scope(cnnLayerName+"-pool"):
       P = tf.nn.max_pool(H, ksize=[1, cnnPoolSizeX,cnnPoolSizeY, 1],strides=[1, cnnPoolSizeX,cnnPoolSizeY , 1], padding='SAME')
       ## put the output of this layer to the next layer's input layer.
       previous_level_convolution_output=P
       self.logger.info(cnnLayerName+".H_pooled.shape="+str(P.shape))
     else :
       ## no residual for layer liner CNN as fourier transform.
       previous_level_convolution_output=H

     previous_level_kernel_count=cnnKernelCount
     last_layer_output=previous_level_convolution_output

   logger.info("self.last_layer_output.shape="+str(last_layer_output.shape))
##############################################################################################
#BURADAR LINERAIZATION GEREKECEK
##############################################################################################
   with tf.name_scope('cnn_to_fc_reshape'):
    last_layer_output_flat = tf.reshape(last_layer_output, [-1, int(last_layer_output.shape[1]*last_layer_output.shape[2]*last_layer_output.shape[3])] )
    self.logger.info("last_layer_output_flat="+str(last_layer_output_flat))
    last_layer_output=last_layer_output_flat


   # x_input.shape[4]=frequency_slice_length+ number_of_frequency_slices * (frequency_slice_length-frequency_slice_overlap_length) 
   a=last_layer_output.shape[1]-self.frequency_slice_length
   b=self.frequency_slice_length-self.frequency_slice_overlap_length
   self.number_of_frequency_slices=int(int(a)/int(b)+1)
    
   with tf.name_scope('input_overlapped_reshape'):
    logger.info("self.last_layer_output.shape="+str(last_layer_output.shape))
    #self.x_input_reshaped = tf.reshape(self.x_input, [self.mini_batch_size, self.number_of_frequency_slices, int(self.input_size/self.number_of_frequency_slices)])

    # You can use tf.nn.conv2d to help. Basically, you take a sliding filter of block_size over the input, 
    # stepping by stride. To make all the matrix indexes line up, you have to do some reshaping.    
    #def overlap(tensor, block_size=3, stride=2):
    stride=self.frequency_slice_length-frequency_slice_overlap_length
    reshaped = tf.reshape(last_layer_output, [self.mini_batch_size,1,-1,1])
    
    ones = tf.ones(self.frequency_slice_length, dtype=tf.float32)
    ident = tf.diag(ones)
    filter_dim = [1, self.frequency_slice_length, self.frequency_slice_length, 1]
    filter_matrix = tf.reshape(ident, filter_dim)
    stride_window = [1, 1, stride, 1]
    filtered_conv = []
    for f in tf.unstack(filter_matrix, axis=1):
      reshaped_filter = tf.reshape(f, [1, self.frequency_slice_length, 1, 1])
      c = tf.nn.conv2d(reshaped, reshaped_filter, stride_window, padding='VALID')
      filtered_conv.append(c)
    t = tf.stack(filtered_conv, axis=3)
    self.x_input_reshaped = tf.squeeze(t)
    
    
    logger.info("self.mini_batch_size="+str(self.mini_batch_size))
    logger.info("self.number_of_frequency_slices="+str(self.number_of_frequency_slices))
    logger.info("self.frequency_slice_length="+str(self.frequency_slice_length))
    logger.info("self.input_size_y="+str(self.input_size_y))
    logger.info("number_of_input_channels="+str(number_of_input_channels))
    
    self.x_input_reshaped = tf.reshape(self.x_input_reshaped, [self.mini_batch_size,self.number_of_frequency_slices,
                                                               self.frequency_slice_length,
                                                               self.input_size_y,number_of_input_channels])
    
    #def overlapping_blocker(tensor,block_size=3,stride=2):   
    #self.x_input_reshaped = tf.reshape(self.x_input, [self.mini_batch_size, self.number_of_frequency_slices, int(self.input_size/self.number_of_frequency_slices)])
    # Unstack to get a list of 'number_of_frequency_slices' tensors of shape (batch_size, input_size/number_of_frequency_slices,number_of_input_channels)


   logger.info("self.x_input_reshaped.shape="+str(self.x_input_reshaped.shape))



    
   ##
   ##  CNN LAYERS FOR EACH TIME SLICE 
   ##
   previous_level_convolution_output=self.x_input_reshaped
   with tf.name_scope('cnn_layers'):
    for cnnLayerNo in range(len(self.cnn_kernel_counts)) :
     cnnLayerName    = "cnn-"+str(cnnLayerNo)     
     cnnKernelCount  = self.cnn_kernel_counts[cnnLayerNo]   # cnnKernelCount tane cnnKernelSizeX * cnnKernelSizeY lik convolution kernel uygulanacak , sonucta 64x1x88200 luk tensor cikacak.
     cnnKernelSizeTimeSlice  = self.cnn_kernel_timeslice_sizes[cnnLayerNo]
     cnnKernelSizeX  = self.cnn_kernel_x_sizes[cnnLayerNo]
     cnnKernelSizeY  = self.cnn_kernel_y_sizes[cnnLayerNo]         
     cnnStrideSizeTimeSlice  = self.cnn_stride_timeslice_sizes[cnnLayerNo] 
     cnnStrideSizeX  = self.cnn_stride_x_sizes[cnnLayerNo] 
     cnnStrideSizeY  = self.cnn_stride_y_sizes[cnnLayerNo]                     
     cnnPoolSizeTimeSlice    = self.cnn_pool_timeslice_sizes[cnnLayerNo]          
     cnnPoolSizeX    = self.cnn_pool_x_sizes[cnnLayerNo]          
     cnnPoolSizeY    = self.cnn_pool_y_sizes[cnnLayerNo]      
     cnnOutputChannel= cnnKernelCount   
     if cnnLayerNo == 0 :
       cnnInputChannel = 1
     else :
       cnnInputChannel = self.cnn_kernel_counts[int(cnnLayerNo-1)]   
     with tf.name_scope(cnnLayerName+"-convolution"):
       ## WEIGHT    
       W = tf.Variable(tf.truncated_normal([cnnKernelSizeTimeSlice,cnnKernelSizeX, cnnKernelSizeY, cnnInputChannel, cnnOutputChannel], stddev=0.1))
       self.logger.info(W)
       ## BIAS
       B = tf.Variable(tf.constant(0.1, shape=[cnnOutputChannel]))
       #Based on conv2d doc:
       #    shape of input = [batch, in_height, in_width, in_channels]
       #    shape of filter = [filter_height, filter_width, in_channels, out_channels]
       #    Last dimension of input and third dimension of filter represents the number of input channels
       C = tf.nn.conv3d(previous_level_convolution_output,W,strides=[cnnStrideSizeTimeSlice,cnnStrideSizeX, cnnStrideSizeY, 1, 1], padding='SAME')+B
       self.logger.info(cnnLayerName+"_C.shape="+str(C.shape))
     self.logger.info(C)
     with tf.name_scope(cnnLayerName+"-relu"):  
       H = tf.nn.relu(C)
       self.logger.info(cnnLayerName+"_H.shape="+str(H.shape))

     with tf.name_scope(cnnLayerName+"-pool"):
       # Max Pooling layer - downsamples by pool_length.
       # pooled by 1 x cnnPoolSize
       P = tf.nn.max_pool3d(H, ksize=[1,cnnPoolSizeTimeSlice, cnnPoolSizeX,cnnPoolSizeY, 1],strides=[1, cnnPoolSizeTimeSlice,cnnPoolSizeX,cnnPoolSizeY, 1], padding='SAME') 
       self.logger.info(cnnLayerName+".H_pooled.shape="+str(P.shape))
       ## put the output of this layer to the next layer's input layer.
       previous_level_convolution_output=P
       self.logger.info(P)
    cnn_last_layer_output=previous_level_convolution_output
    
    ## cnn_last_layer_output.shape[0]==self.mini_batch_size
    ## cnn_last_layer_output.shape[1]==self.number_of_frequency_slices
    cnn_last_layer_output=tf.reshape(cnn_last_layer_output, [self.mini_batch_size, 
                                     self.number_of_frequency_slices,cnn_last_layer_output.shape[2]*cnn_last_layer_output.shape[3]*cnn_last_layer_output.shape[4]])
    self.logger.info("cnn_last_layer_output.shape="+str(cnn_last_layer_output.shape))
    self.cnn_last_layer_output_as_list = tf.unstack(cnn_last_layer_output, self.number_of_frequency_slices, 1) ## 0,1,2.. dimensionlar icinde 1. dimension icin unstack et.
    self.logger.info("len(self.cnn_last_layer_output_as_list)="+str(len(self.cnn_last_layer_output_as_list)))
    self.logger.info("self.cnn_last_layer_output_as_list[0].shape"+str(self.cnn_last_layer_output_as_list[0].shape))

   ## LSTM LAYERS
   with tf.name_scope("lstm_layes"):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_state_size, forget_bias=self.lstm_forget_bias)
    lstm_cell_with_dropout=tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob_constant)
    lstm_outputs, lstm_state = tf.nn.static_rnn(lstm_cell_with_dropout, inputs=self.cnn_last_layer_output_as_list, dtype=tf.float32)
    self.logger.info("lstm_outputs="+str( lstm_outputs))
    self.logger.info("lstm_state="+str( lstm_state))
#    weights = {'out': tf.Variable(tf.random_normal([self.lstm_state_size, self.output_size ]))}
#    biases = {'out': tf.Variable(tf.random_normal([self.output_size ]))}
#    self.y_outputs=tf.matmul(lstm_outputs[-1], weights['out']) + biases['out']

   ##
   ## FULLY CONNECTED LAYERS
   ##Linear activation (FC layer on top of the LSTM net)

   with tf.name_scope('fc1'):
    W_fc1 =  tf.Variable( tf.truncated_normal([self.lstm_state_size, self.number_of_fully_connected_layer_neurons], stddev=0.1))
    self.logger.info("W_fc1.shape="+str(W_fc1.shape))
    B_fc1 = tf.Variable(tf.constant(0.2, shape=[self.number_of_fully_connected_layer_neurons]))
    self.logger.info("B_fc1.shape="+str(B_fc1.shape))
    matmul_fc1=tf.matmul(lstm_outputs[-1], W_fc1)+B_fc1
    self.logger.info("matmul_fc1.shape="+str(matmul_fc1.shape))
    
   with tf.name_scope('fc1_batch_normlalization'):    
    batch_mean, batch_var = tf.nn.moments(matmul_fc1,[0])
    scale = tf.Variable(tf.ones(self.number_of_fully_connected_layer_neurons))
    beta = tf.Variable(tf.zeros(self.number_of_fully_connected_layer_neurons))
    batch_normalization_fc1 = tf.nn.batch_normalization(matmul_fc1,batch_mean,batch_var,beta,scale,epsilon)
    self.logger.info("batch_normalization_fc1.shape="+str(batch_normalization_fc1.shape))

   with tf.name_scope('fc1_batch_normalized_relu'):    
    h_fc1 = tf.nn.relu( batch_normalization_fc1 )
    self.logger.info("h_fc1.shape="+str(h_fc1.shape))

   # Dropout - controls the complexity of the model, prevents co-adaptation of features.
   with tf.name_scope('fc1_dropout'):    
    self.keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
    self.logger.info("h_fc1_drop.shape="+str(h_fc1_drop.shape))

    # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS(1024) features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('fc2'):
    W_fc2 =  tf.Variable( tf.truncated_normal([self.number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs =tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    self.logger.info("self.y_outputs.shape="+str(self.y_outputs.shape))
      
   ## HERE NETWORK DEFINITION IS FINISHED
    
#    ##  NOW CALCULATE PREDICTED VALUE
#    with tf.name_scope('calculate_predictions'):
#     output_shape = tf.shape(self.y_outputs)
#     self.predictions = tf.nn.softmax(tf.reshape(self.y_outputs, [-1, self.output_size]))
     
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
    self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

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
   self.session.run(tf.global_variables_initializer())

 def prepareData(self,data,augment):
  x_data=data[:,:4*SOUND_RECORD_SAMPLING_RATE]
  if augment==True :
    x_data=augment_random(x_data)
  y_data=data[:,4*SOUND_RECORD_SAMPLING_RATE]
  y_data_one_hot_encoded=one_hot_encode_array(y_data)
  return x_data,y_data_one_hot_encoded

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


 



