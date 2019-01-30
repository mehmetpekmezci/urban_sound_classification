#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class NeuralNetworkModel :
 def __init__(self, session, logger, 
              input_size=INPUT_SIZE, output_size=OUTPUT_SIZE , 
              learning_rate=LEARNING_RATE, mini_batch_size=MINI_BATCH_SIZE,
              learning_rate_beta1=LEARNING_RATE_BETA1, 
              learning_rate_beta2=LEARNING_RATE_BETA2, 
              epsilon=EPSILON,keep_prob_constant=KEEP_PROB,
              fully_connected_layers=FULLY_CONNECTED_LAYERS,
              fourier_fully_connected_layers=FOURIER_FULLY_CONNECTED_LAYERS):

   ##
   ## SET CLASS ATTRIBUTES WITH THE GIVEN INPUTS
   ##
   self.session               = session
   self.logger                = logger
   self.input_size            = input_size
   self.input_size_y          = 1
   self.output_size           = output_size
   self.cnn_kernel_counts     = cnn_kernel_counts
   self.cnn_kernel_x_sizes    = cnn_kernel_x_sizes
   self.cnn_kernel_y_sizes    = cnn_kernel_y_sizes
   self.cnn_stride_x_sizes    = cnn_stride_x_sizes
   self.cnn_stride_y_sizes    = cnn_stride_y_sizes
   self.cnn_pool_x_sizes      = cnn_pool_x_sizes
   self.cnn_pool_y_sizes      = cnn_pool_y_sizes
   self.learning_rate         = learning_rate 
   self.learning_rate_beta1   = learning_rate_beta1 
   self.learning_rate_beta2   = learning_rate_beta2 
   self.mini_batch_size       = mini_batch_size
   self.keep_prob_constant    = keep_prob_constant
   self.epsilon               = epsilon
   self.fully_connected_layers=fully_connected_layers
   self.fourier_fully_connected_layers=fourier_fully_connected_layers



   self.keep_prob = tf.placeholder(tf.float32)

   

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
    self.x_input_reshaped = tf.reshape(self.x_input, [self.mini_batch_size, self.input_size_y, self.input_size, number_of_input_channels])
    print("self.x_input_reshaped.shape="+str(self.x_input_reshaped.shape))
    previous_level_convolution_output = self.x_input_reshaped
    last_layer_output=previous_level_convolution_output

   for fourierFcLayerNo in range(len(self.fourier_fully_connected_layers)) :
       
    number_of_fourier_fully_connected_layer_neurons=self.fourier_fully_connected_layers[fourierFcLayerNo]

    with tf.name_scope('fourier-fc-'+str(fourierFcLayerNo)):
     W_fc1 =  tf.Variable( tf.truncated_normal([int(last_layer_output.shape[1]), number_of_fourier_fully_connected_layer_neurons], stddev=0.1))
     self.logger.info("fourier-W_fc-"+str(fourierFcLayerNo)+".shape="+str(W_fc1.shape))
     B_fc1 = tf.Variable(tf.constant(0.1, shape=[number_of_fourier_fully_connected_layer_neurons]))
     self.logger.info("fourier-B_fc-"+str(fourierFcLayerNo)+".shape="+str(B_fc1.shape))
     matmul_fc1=tf.matmul(last_layer_output, W_fc1)+B_fc1
     self.logger.info("fourier-matmul_fc-"+str(fourierFcLayerNo)+".shape="+str(matmul_fc1.shape))

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
     h_fc1_drop = tf.nn.dropout(matmul_fc1, self.keep_prob)
     self.logger.info("h_fc-"+str(fourierFcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
     last_layer_output=h_fc1_drop



   
   ##
   ## FULLY CONNECTED LAYERS
   ##Linear activation (FC layer on top of the RESNET )
   number_of_fully_connected_layer_neurons=self.fully_connected_layers[0]
   
   for fcLayerNo in range(len(self.fully_connected_layers)) :
       
    number_of_fully_connected_layer_neurons=self.fully_connected_layers[fcLayerNo]

    with tf.name_scope('fc-'+str(fcLayerNo)):
     W_fc1 =  tf.Variable( tf.truncated_normal([int(last_layer_output.shape[1]), number_of_fully_connected_layer_neurons], stddev=0.1))
     self.logger.info("W_fc-"+str(fcLayerNo)+".shape="+str(W_fc1.shape))
     B_fc1 = tf.Variable(tf.constant(0.1, shape=[number_of_fully_connected_layer_neurons]))
     self.logger.info("B_fc-"+str(fcLayerNo)+".shape="+str(B_fc1.shape))
     matmul_fc1=tf.matmul(last_layer_output, W_fc1)+B_fc1
     self.logger.info("matmul_fc-"+str(fcLayerNo)+".shape="+str(matmul_fc1.shape))

    with tf.name_scope('fc-'+str(fcLayerNo)+'_batch_normlalization'):    
     batch_mean, batch_var = tf.nn.moments(matmul_fc1,[0])
     scale = tf.Variable(tf.ones(number_of_fully_connected_layer_neurons))
     beta = tf.Variable(tf.zeros(number_of_fully_connected_layer_neurons))
     batch_normalization_fc1 = tf.nn.batch_normalization(matmul_fc1,batch_mean,batch_var,beta,scale,epsilon)
     self.logger.info("batch_normalization_fc-"+str(fcLayerNo)+".shape="+str(batch_normalization_fc1.shape))

    with tf.name_scope('fc-'+str(fcLayerNo)+'_batch_normalized_relu'):    
     h_fc1 = tf.nn.relu( batch_normalization_fc1 )
     self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
     h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
     self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
     last_layer_output=h_fc1_drop


   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs.shape="+str(self.y_outputs.shape))
      
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
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.learning_rate_beta1,beta2=self.learning_rate_beta2).minimize(self.loss)

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
  



