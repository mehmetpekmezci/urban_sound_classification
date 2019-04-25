#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## DENIOSING AUTO ENCODER
##
class AutoEncoder :
 def __init__(self,session,logger,input_size=INPUT_SIZE,learning_rate=LEARNING_RATE,mini_batch_size=MINI_BATCH_SIZE,encoder_layers=ENCODER_LAYERS,keep_prob_constant=KEEP_PROB,epsilon=EPSILON):

   ##
   ## SET CLASS ATTRIBUTES WITH THE GIVEN INPUTS
   ##
   self.session               = session
   self.logger                = logger
   self.input_size            = input_size
   self.learning_rate         = learning_rate 
   self.mini_batch_size       = mini_batch_size
   self.encoder_layers        = encoder_layers
   self.keep_prob_constant    = keep_prob_constant
   self.epsilon               = epsilon  
  
   self.keep_prob = tf.placeholder(tf.float32)

   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER
   ##
   self.x_noisy_input                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   self.x_clean_input                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   
   last_layer_output=self.x_noisy_input
   with tf.name_scope('autoencoder'):
    for fcLayerNo in range(len(self.encoder_layers)) :
       number_of_fully_connected_layer_neurons=self.encoder_layers[fcLayerNo]
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
   with tf.name_scope('output_autocoder'):
     W_fc2 =  tf.Variable( tf.truncated_normal([int(last_layer_output.shape[1]), self.input_size], stddev=0.1))
     b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.input_size]))
     self.y_output =tf.matmul(last_layer_output, W_fc2) + b_fc2
     self.logger.info("self.y_output.shape="+str(self.y_output.shape))
      
    ## HERE NETWORK DEFINITION IS FINISHED
     
   ##
   ## CALCULATE LOSS
   ##
   with tf.name_scope('calculate_loss'):
     self.loss=tf.losses.mean_squared_error(self.x_clean_input,self.y_output)

   ##
   ## SET OPTIMIZER
   ##
   with tf.name_scope('optimizer'):
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

   ##
   ## SAVE NETWORK GRAPH TO A DIRECTORY
   ##
   with tf.name_scope('save_graph'):
    self.logger.info('Saving graph to: %s-autoencoder' % LOG_DIR_FOR_TF_SUMMARY)
    graph_writer = tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY+"-autoencoder")
    graph_writer.add_graph(tf.get_default_graph())

 def prepareData(self,data):
  x_data=augment_random(data[:,:4*SOUND_RECORD_SAMPLING_RATE])
  x_noisy_data=x_data+generate_normalized_synthetic_noise(x_data.shape[0])
  x_clean_data=np.random.permutation(x_data)
  x_noisy_data=np.random.permutation(x_noisy_data)
  return x_clean_data,x_noisy_data

 def train(self,data):
  prepareDataTimeStart = int(round(time.time())) 
  x_clean_data,x_noisy_data=self.prepareData(data)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  self.optimizer.run(feed_dict={self.x_clean_input: x_clean_data,self.x_noisy_input: x_noisy_data,self.keep_prob: self.keep_prob_constant})
  ######
  ####  INPUt=noisy
  ###   OUTPUT=clean
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingLoss = self.loss.eval(feed_dict={self.x_clean_input: x_clean_data,self.x_noisy_input: x_noisy_data,self.keep_prob: 1.0})
  return trainingTime,trainingLoss,prepareDataTime
     
 def denoise(self,data):
  encodeTimeStart = int(round(time.time())) 
  x_data=data[:,:4*SOUND_RECORD_SAMPLING_RATE]
  denoisedValue = self.y_output.eval(feed_dict={self.x_noisy_input: x_data, self.keep_prob: 1.0})
 # encodedValue = self.encoder.eval(feed_dict={self.x_input: x_data, self.keep_prob: 1.0})
  encodeTimeStop = int(round(time.time())) 
  encodeTime=encodeTimeStop-encodeTimeStart
  return denoisedValue,encodeTime
  
