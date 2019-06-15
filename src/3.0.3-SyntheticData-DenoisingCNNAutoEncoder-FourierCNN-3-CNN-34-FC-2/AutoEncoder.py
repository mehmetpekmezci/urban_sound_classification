#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## DENIOSING AUTO ENCODER
##
class AutoEncoder :
 def __init__(self,session,logger,input_size=INPUT_SIZE,learning_rate=LEARNING_RATE,mini_batch_size=MINI_BATCH_SIZE,
              cnn_kernel_counts=AE_CNN_KERNEL_COUNTS,
              cnn_kernel_x_sizes=AE_CNN_KERNEL_X_SIZES,cnn_kernel_y_sizes=AE_CNN_KERNEL_Y_SIZES,
              cnn_stride_x_sizes=AE_CNN_STRIDE_X_SIZES,cnn_stride_y_sizes=AE_CNN_STRIDE_Y_SIZES,
              cnn_pool_x_sizes=AE_CNN_POOL_X_SIZES,cnn_pool_y_sizes=AE_CNN_POOL_Y_SIZES,
              encoder_layers=ENCODER_LAYERS,keep_prob_constant=KEEP_PROB,epsilon=EPSILON):

   ##
   ## SET CLASS ATTRIBUTES WITH THE GIVEN INPUTS
   ##
   self.session               = session
   self.logger                = logger
   self.input_size_y          = 1
   self.input_size            = input_size
   self.learning_rate         = learning_rate 
   self.mini_batch_size       = mini_batch_size
   self.encoder_layers        = encoder_layers
   self.keep_prob_constant    = keep_prob_constant
   self.epsilon               = epsilon  
   self.cnn_kernel_counts     = cnn_kernel_counts
   self.cnn_kernel_x_sizes    = cnn_kernel_x_sizes
   self.cnn_kernel_y_sizes    = cnn_kernel_y_sizes
   self.cnn_stride_x_sizes    = cnn_stride_x_sizes
   self.cnn_stride_y_sizes    = cnn_stride_y_sizes
   self.cnn_pool_x_sizes      = cnn_pool_x_sizes
   self.cnn_pool_y_sizes      = cnn_pool_y_sizes



   self.keep_prob = tf.placeholder(tf.float32)



   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER
   ##
   number_of_input_channels=1
   self.x_noisy_input                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   self.x_clean_input                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   
   with tf.name_scope('input_reshape'):
     self.x_input_reshaped = tf.reshape(last_layer_output, [self.mini_batch_size, self.input_size_y, last_layer_output.shape[1], number_of_input_channels])
     self.logger.info("self.x_input_reshaped.shape="+str(self.x_input_reshaped.shape))

   previous_level_convolution_output = self.x_input_reshaped
   for cnnLayerNo in range(len(self.cnn_kernel_counts)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
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
       W = tf.Variable(tf.truncated_normal([cnnKernelSizeX, cnnKernelSizeY, cnnInputChannel, cnnOutputChannel], stddev=0.1))
       B = tf.Variable(tf.constant(0.1, shape=[cnnOutputChannel]))
       C = tf.nn.conv2d(previous_level_convolution_output,W,strides=[1,cnnStrideSizeX, cnnStrideSizeY, 1], padding='SAME')+B

       self.logger.info(cnnLayerName+"_C.shape="+str(C.shape)+"  W.shape="+str(W.shape)+ "  cnnStrideSizeX="+str(cnnStrideSizeX)+" cnnStrideSizeY="+str(cnnStrideSizeY))
     with tf.name_scope(cnnLayerName+"-relu"):
       H = tf.nn.relu(C)
       self.logger.info(cnnLayerName+"_H.shape="+str(H.shape))

     if cnnPoolSizeY != 1 :
      with tf.name_scope(cnnLayerName+"-pool"):
       P = tf.nn.max_pool(H, ksize=[1, cnnPoolSizeX,cnnPoolSizeY, 1],strides=[1, cnnPoolSizeX,cnnPoolSizeY , 1], padding='SAME')
       ## put the output of this layer to the next layer's input layer.
       previous_level_convolution_output=P
       self.logger.info(cnnLayerName+".H_pooled.shape="+str(P.shape))
     else :
      if previous_level_kernel_count==cnnKernelCount :
       with tf.name_scope(cnnLayerName+"-residual"):
         previous_level_convolution_output=H+previous_level_convolution_output
         ## put the output of this layer to the next layer's input layer.
         self.logger.info(cnnLayerName+"_previous_level_convolution_output_residual.shape="+str(previous_level_convolution_output.shape))
      else :
         ## put the output of this layer to the next layer's input layer.
         previous_level_convolution_output=H

     previous_level_kernel_count=cnnKernelCount
     cnn_last_layer_output=previous_level_convolution_output


   ##
   ## FULLY CONNECTED LAYERS
   ##Linear activation (FC layer on top of the RESNET )


   with tf.name_scope('cnn_to_fc_reshape'):
    cnn_last_layer_output_flat = tf.reshape( cnn_last_layer_output, [-1, int(cnn_last_layer_output.shape[1]*cnn_last_layer_output.shape[2]*cnn_last_layer_output.shape[3])] )
    self.logger.info("cnn_last_layer_output_flat="+str( cnn_last_layer_output_flat))

   last_layer_output=cnn_last_layer_output_flat
   encoder=last_layer_output

   ### DECODER
   with tf.name_scope('decoder'):
    for fcLayerNo in range(len(self.encoder_layers)) :
       number_of_fully_connected_layer_neurons=self.encoder_layers[int(-1*(fcLayerNo+1))]
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

   ### OUTPUT
   with tf.name_scope('output_decoder'):
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
  
