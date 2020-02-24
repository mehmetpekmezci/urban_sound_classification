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
              cnn_kernel_counts=CNN_KERNEL_COUNTS, 
              cnn_kernel_x_sizes=CNN_KERNEL_X_SIZES,cnn_kernel_y_sizes=CNN_KERNEL_Y_SIZES,
              cnn_stride_x_sizes=CNN_STRIDE_X_SIZES,cnn_stride_y_sizes=CNN_STRIDE_Y_SIZES,
              cnn_pool_x_sizes=CNN_POOL_X_SIZES,cnn_pool_y_sizes=CNN_POOL_Y_SIZES, 
              learning_rate=LEARNING_RATE, mini_batch_size=MINI_BATCH_SIZE,
              learning_rate_beta1=LEARNING_RATE_BETA1, 
              learning_rate_beta2=LEARNING_RATE_BETA2, 
              epsilon=EPSILON,keep_prob_constant=KEEP_PROB,
              fully_connected_layers=FULLY_CONNECTED_LAYERS,
              fourier_cnn_kernel_counts=FOURIER_CNN_KERNEL_COUNTS, 
              fourier_cnn_kernel_x_sizes=FOURIER_CNN_KERNEL_X_SIZES,fourier_cnn_kernel_y_sizes=FOURIER_CNN_KERNEL_Y_SIZES,
              fourier_cnn_stride_x_sizes=FOURIER_CNN_STRIDE_X_SIZES,fourier_cnn_stride_y_sizes=FOURIER_CNN_STRIDE_Y_SIZES,
              fourier_cnn_pool_x_sizes=FOURIER_CNN_POOL_X_SIZES,fourier_cnn_pool_y_sizes=FOURIER_CNN_POOL_Y_SIZES, 
              metric_cnn_kernel_counts=FOURIER_CNN_KERNEL_COUNTS, 
              metric_cnn_kernel_x_sizes=FOURIER_CNN_KERNEL_X_SIZES,metric_cnn_kernel_y_sizes=FOURIER_CNN_KERNEL_Y_SIZES,
              metric_cnn_stride_x_sizes=FOURIER_CNN_STRIDE_X_SIZES,metric_cnn_stride_y_sizes=FOURIER_CNN_STRIDE_Y_SIZES,
              metric_cnn_pool_x_sizes=FOURIER_CNN_POOL_X_SIZES,metric_cnn_pool_y_sizes=FOURIER_CNN_POOL_Y_SIZES, 
              metric_fully_connected_layers=METRIC_FULLY_CONNECTED_LAYERS,              
              cut_into_parts_number=CUT_INTO_PARTS_NUMBER
              ):

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
   self.fourier_cnn_kernel_counts     = fourier_cnn_kernel_counts
   self.fourier_cnn_kernel_x_sizes    = fourier_cnn_kernel_x_sizes
   self.fourier_cnn_kernel_y_sizes    = fourier_cnn_kernel_y_sizes
   self.fourier_cnn_stride_x_sizes    = fourier_cnn_stride_x_sizes
   self.fourier_cnn_stride_y_sizes    = fourier_cnn_stride_y_sizes
   self.fourier_cnn_pool_x_sizes      = fourier_cnn_pool_x_sizes
   self.fourier_cnn_pool_y_sizes      = fourier_cnn_pool_y_sizes
   
   self.metric_fully_connected_layers=metric_fully_connected_layers
   self.metric_cnn_kernel_counts     = metric_cnn_kernel_counts
   self.metric_cnn_kernel_x_sizes    = metric_cnn_kernel_x_sizes
   self.metric_cnn_kernel_y_sizes    = metric_cnn_kernel_y_sizes
   self.metric_cnn_stride_x_sizes    = metric_cnn_stride_x_sizes
   self.metric_cnn_stride_y_sizes    = metric_cnn_stride_y_sizes
   self.metric_cnn_pool_x_sizes      = metric_cnn_pool_x_sizes
   self.metric_cnn_pool_y_sizes      = metric_cnn_pool_y_sizes
   self.cut_into_parts_number        = cut_into_parts_number
   self.ADVERSERIAL_TRESHOLD=0.5

   self.keep_prob = tf.placeholder(tf.float32)

   

   ##
   ## DEFINE PLACE HOLDER FOR REAL OUTPUT VALUES FOR TRAINING
   ##
   self.real_y_values_1=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values_1")
   self.real_y_values_2=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values_2")
   self.real_y_values_3=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values_3")
   self.real_y_values_adverserial_1_2=tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1), name="real_y_values_adverserial_1_2")
   self.real_y_values_adverserial_1_3=tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1), name="real_y_values_adverserial_1_3")
   self.real_y_values_adverserial_2_3=tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1), name="real_y_values_adverserial_2_3")

   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER
   ##
   number_of_input_channels=1
   self.x_input_1                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input_1")
   self.x_input_2                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input_2")
   self.x_input_3                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input_3")
   last_layer_output_1=self.x_input_1
   last_layer_output_2=self.x_input_2
   last_layer_output_3=self.x_input_3

   ##
   ## RESHAPE
   ##
   with tf.name_scope('input_reshape'):
     self.x_input_reshaped_1 = tf.reshape(last_layer_output_1, [self.mini_batch_size, self.input_size_y, int(last_layer_output_1.shape[1]), number_of_input_channels])
     self.x_input_reshaped_2 = tf.reshape(last_layer_output_2, [self.mini_batch_size, self.input_size_y, int(last_layer_output_2.shape[1]), number_of_input_channels])
     self.x_input_reshaped_3 = tf.reshape(last_layer_output_3, [self.mini_batch_size, self.input_size_y, int(last_layer_output_3.shape[1]), number_of_input_channels])
     self.logger.info("self.x_input_reshaped_1.shape="+str(self.x_input_reshaped_1.shape))
     self.logger.info("self.x_input_reshaped_2.shape="+str(self.x_input_reshaped_2.shape))
     self.logger.info("self.x_input_reshaped_3.shape="+str(self.x_input_reshaped_3.shape))
     previous_level_convolution_output_1 = self.x_input_reshaped_1
     previous_level_convolution_output_2 = self.x_input_reshaped_2
     previous_level_convolution_output_3 = self.x_input_reshaped_3

   previous_level_convolution_output=tf.concat((previous_level_convolution_output_1,previous_level_convolution_output_2,previous_level_convolution_output_3),1)


   ## RESHAPE TO 2 x cut_into_parts_number

   previous_level_convolution_output=tf.reshape(previous_level_convolution_output, [-1,int(int(previous_level_convolution_output.shape[1])*self.cut_into_parts_number),int(int(previous_level_convolution_output.shape[2])/self.cut_into_parts_number), int(previous_level_convolution_output.shape[3])])

   ##
   ## FOURIER  CNN LAYERS
   ##
   with tf.name_scope('fourier_CNN'):
    for fourierCNNLayerNo in range(len(self.fourier_cnn_kernel_counts)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "fourier-cnn-"+str(fourierCNNLayerNo)          
     cnnKernelCount  = self.fourier_cnn_kernel_counts[fourierCNNLayerNo]  
     cnnKernelSizeX  = self.fourier_cnn_kernel_x_sizes[fourierCNNLayerNo]
     cnnKernelSizeY  = self.fourier_cnn_kernel_y_sizes[fourierCNNLayerNo]         
     cnnStrideSizeX  = self.fourier_cnn_stride_x_sizes[fourierCNNLayerNo] 
     cnnStrideSizeY  = self.fourier_cnn_stride_y_sizes[fourierCNNLayerNo]                     
     cnnPoolSizeX    = self.fourier_cnn_pool_x_sizes[fourierCNNLayerNo]          
     cnnPoolSizeY    = self.fourier_cnn_pool_y_sizes[fourierCNNLayerNo]      
     
     
     cnnOutputChannel= cnnKernelCount   
     if fourierCNNLayerNo == 0 :
       cnnInputChannel = 1
     else :
       cnnInputChannel = self.fourier_cnn_kernel_counts[int(fourierCNNLayerNo-1)]   


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

   ##
   ## CNN LAYERS
   ##

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
       cnnInputChannel = int (previous_level_convolution_output.shape[3])
       previous_level_kernel_count=1
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
    cnn_last_layer_output_1 = cnn_last_layer_output[:,0,:,:]
    cnn_last_layer_output_2 = cnn_last_layer_output[:,1,:,:]
    cnn_last_layer_output_3 = cnn_last_layer_output[:,2,:,:]
    cnn_last_layer_output_flat_1 = tf.reshape( cnn_last_layer_output_1, [-1, int(cnn_last_layer_output_1.shape[1]*cnn_last_layer_output_1.shape[2])] )
    cnn_last_layer_output_flat_2 = tf.reshape( cnn_last_layer_output_2, [-1, int(cnn_last_layer_output_2.shape[1]*cnn_last_layer_output_2.shape[2])] )
    cnn_last_layer_output_flat_3 = tf.reshape( cnn_last_layer_output_3, [-1, int(cnn_last_layer_output_3.shape[1]*cnn_last_layer_output_3.shape[2])] )
    self.logger.info("cnn_last_layer_output_flat_1="+str( cnn_last_layer_output_flat_1))
    self.logger.info("cnn_last_layer_output_flat_2="+str( cnn_last_layer_output_flat_2))
    self.logger.info("cnn_last_layer_output_flat_3="+str( cnn_last_layer_output_flat_3))


   ##
   ## DOUBLE OUTPUT
   ##


   ## FISRT FULLY CONNECTED 
   last_layer_output=cnn_last_layer_output_flat_1
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
     h_fc1 = tf.nn.sigmoid( batch_normalization_fc1 )
     self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
     h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
     self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
     last_layer_output=h_fc1_drop
     first_fc_output=last_layer_output


   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs_1 =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs_1.shape="+str(self.y_outputs_1.shape))
      


   ## SECOND FULLY CONNECTED 

   last_layer_output=cnn_last_layer_output_flat_2
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
     h_fc1 = tf.nn.sigmoid( batch_normalization_fc1 )
     self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
     h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
     self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
     last_layer_output=h_fc1_drop
     second_fc_output=last_layer_output

   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs_2 =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs_2.shape="+str(self.y_outputs_2.shape))
    

   ## THIRD FULLY CONNECTED 

   last_layer_output=cnn_last_layer_output_flat_3
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
     h_fc1 = tf.nn.sigmoid( batch_normalization_fc1 )
     self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
     h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
     self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
     last_layer_output=h_fc1_drop
     second_fc_output=last_layer_output

   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs_3 =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs_3.shape="+str(self.y_outputs_3.shape))
    






   # DISCRIMINATOR 1_2

   ##
   ## METRIC  CNN LAYERS
   ##
   previous_level_convolution_output=cnn_last_layer_output

   with tf.name_scope('metric_CNN'):
    for metricCNNLayerNo in range(len(self.metric_cnn_kernel_counts)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "metric-cnn-"+str(metricCNNLayerNo)          
     cnnKernelCount  = self.metric_cnn_kernel_counts[metricCNNLayerNo]  
     cnnKernelSizeX  = self.metric_cnn_kernel_x_sizes[metricCNNLayerNo]
     cnnKernelSizeY  = self.metric_cnn_kernel_y_sizes[metricCNNLayerNo]         
     cnnStrideSizeX  = self.metric_cnn_stride_x_sizes[metricCNNLayerNo] 
     cnnStrideSizeY  = self.metric_cnn_stride_y_sizes[metricCNNLayerNo]                     
     cnnPoolSizeX    = self.metric_cnn_pool_x_sizes[metricCNNLayerNo]          
     cnnPoolSizeY    = self.metric_cnn_pool_y_sizes[metricCNNLayerNo]      
     
     
     cnnOutputChannel= cnnKernelCount   
     if metricCNNLayerNo == 0 :
       cnnInputChannel = int (previous_level_convolution_output.shape[3])
       previous_level_kernel_count=1
       #cnnInputChannel = 1
     else :
       cnnInputChannel = self.metric_cnn_kernel_counts[int(metricCNNLayerNo-1)]   


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
       ## no residual for layer liner CNN as fourier transform.
       previous_level_convolution_output=H
     previous_level_kernel_count=cnnKernelCount



   metric_cnn_last_layer_output=previous_level_convolution_output
   last_layer_output=tf.reshape( metric_cnn_last_layer_output, [-1, int(metric_cnn_last_layer_output.shape[1]*metric_cnn_last_layer_output.shape[2]*metric_cnn_last_layer_output.shape[3])] )

   #for fcLayerNo in range(len(self.fully_connected_layers)) :
       
   number_of_fully_connected_layer_neurons=self.metric_fully_connected_layers[0]

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
    h_fc1 = tf.nn.sigmoid( batch_normalization_fc1 )
    self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))
    last_layer_output=h_fc1

   # Dropout - controls the complexity of the model, prevents co-adaptation of features.
   #with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
   # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
   # self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
   # last_layer_output=h_fc1_drop

   #adverserial output=0/1  yes or no, meaning that these two outputs are the same or not
   adverserial_output_size=1
   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(1) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, adverserial_output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[adverserial_output_size]))
    self.y_outputs_adverserial_1_2 =tf.nn.sigmoid(tf.matmul(last_layer_output, W_fc2) + b_fc2)
    self.logger.info("self.y_outputs_adverserial_1_2.shape="+str(self.y_outputs_adverserial_1_2.shape))
    







   # DISCRIMINATOR 1_3

   ##
   ## METRIC  CNN LAYERS
   ##
   previous_level_convolution_output=cnn_last_layer_output

   with tf.name_scope('metric_CNN'):
    for metricCNNLayerNo in range(len(self.metric_cnn_kernel_counts)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "metric-cnn-"+str(metricCNNLayerNo)          
     cnnKernelCount  = self.metric_cnn_kernel_counts[metricCNNLayerNo]  
     cnnKernelSizeX  = self.metric_cnn_kernel_x_sizes[metricCNNLayerNo]
     cnnKernelSizeY  = self.metric_cnn_kernel_y_sizes[metricCNNLayerNo]         
     cnnStrideSizeX  = self.metric_cnn_stride_x_sizes[metricCNNLayerNo] 
     cnnStrideSizeY  = self.metric_cnn_stride_y_sizes[metricCNNLayerNo]                     
     cnnPoolSizeX    = self.metric_cnn_pool_x_sizes[metricCNNLayerNo]          
     cnnPoolSizeY    = self.metric_cnn_pool_y_sizes[metricCNNLayerNo]      
     
     
     cnnOutputChannel= cnnKernelCount   
     if metricCNNLayerNo == 0 :
       cnnInputChannel = int (previous_level_convolution_output.shape[3])
       previous_level_kernel_count=1
       #cnnInputChannel = 1
     else :
       cnnInputChannel = self.metric_cnn_kernel_counts[int(metricCNNLayerNo-1)]   


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
       ## no residual for layer liner CNN as fourier transform.
       previous_level_convolution_output=H
     previous_level_kernel_count=cnnKernelCount



   metric_cnn_last_layer_output=previous_level_convolution_output
   last_layer_output=tf.reshape( metric_cnn_last_layer_output, [-1, int(metric_cnn_last_layer_output.shape[1]*metric_cnn_last_layer_output.shape[2]*metric_cnn_last_layer_output.shape[3])] )

   #for fcLayerNo in range(len(self.fully_connected_layers)) :
       
   number_of_fully_connected_layer_neurons=self.metric_fully_connected_layers[0]

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
    h_fc1 = tf.nn.sigmoid( batch_normalization_fc1 )
    self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))
    last_layer_output=h_fc1

   # Dropout - controls the complexity of the model, prevents co-adaptation of features.
   #with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
   # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
   # self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
   # last_layer_output=h_fc1_drop

   #adverserial output=0/1  yes or no, meaning that these two outputs are the same or not
   adverserial_output_size=1
   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(1) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, adverserial_output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[adverserial_output_size]))
    self.y_outputs_adverserial_1_3 =tf.nn.sigmoid(tf.matmul(last_layer_output, W_fc2) + b_fc2)
    self.logger.info("self.y_outputs_adverserial_1_3.shape="+str(self.y_outputs_adverserial_1_3.shape))
    







   # DISCRIMINATOR 2_3

   ##
   ## METRIC  CNN LAYERS
   ##
   previous_level_convolution_output=cnn_last_layer_output

   with tf.name_scope('metric_CNN'):
    for metricCNNLayerNo in range(len(self.metric_cnn_kernel_counts)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "metric-cnn-"+str(metricCNNLayerNo)          
     cnnKernelCount  = self.metric_cnn_kernel_counts[metricCNNLayerNo]  
     cnnKernelSizeX  = self.metric_cnn_kernel_x_sizes[metricCNNLayerNo]
     cnnKernelSizeY  = self.metric_cnn_kernel_y_sizes[metricCNNLayerNo]         
     cnnStrideSizeX  = self.metric_cnn_stride_x_sizes[metricCNNLayerNo] 
     cnnStrideSizeY  = self.metric_cnn_stride_y_sizes[metricCNNLayerNo]                     
     cnnPoolSizeX    = self.metric_cnn_pool_x_sizes[metricCNNLayerNo]          
     cnnPoolSizeY    = self.metric_cnn_pool_y_sizes[metricCNNLayerNo]      
     
     
     cnnOutputChannel= cnnKernelCount   
     if metricCNNLayerNo == 0 :
       cnnInputChannel = int (previous_level_convolution_output.shape[3])
       previous_level_kernel_count=1
       #cnnInputChannel = 1
     else :
       cnnInputChannel = self.metric_cnn_kernel_counts[int(metricCNNLayerNo-1)]   


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
       ## no residual for layer liner CNN as fourier transform.
       previous_level_convolution_output=H
     previous_level_kernel_count=cnnKernelCount



   metric_cnn_last_layer_output=previous_level_convolution_output
   last_layer_output=tf.reshape( metric_cnn_last_layer_output, [-1, int(metric_cnn_last_layer_output.shape[1]*metric_cnn_last_layer_output.shape[2]*metric_cnn_last_layer_output.shape[3])] )

   #for fcLayerNo in range(len(self.fully_connected_layers)) :
       
   number_of_fully_connected_layer_neurons=self.metric_fully_connected_layers[0]

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
    h_fc1 = tf.nn.sigmoid( batch_normalization_fc1 )
    self.logger.info("h_fc-"+str(fcLayerNo)+".shape="+str(h_fc1.shape))
    last_layer_output=h_fc1

   # Dropout - controls the complexity of the model, prevents co-adaptation of features.
   #with tf.name_scope('fc-'+str(fcLayerNo)+'_dropout'):    
   # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
   # self.logger.info("h_fc-"+str(fcLayerNo)+"_drop.shape="+str(h_fc1_drop.shape))
   # last_layer_output=h_fc1_drop

   #adverserial output=0/1  yes or no, meaning that these two outputs are the same or not
   adverserial_output_size=1
   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(1) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, adverserial_output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[adverserial_output_size]))
    self.y_outputs_adverserial_2_3 =tf.nn.sigmoid(tf.matmul(last_layer_output, W_fc2) + b_fc2)
    self.logger.info("self.y_outputs_adverserial_2_3.shape="+str(self.y_outputs_adverserial_2_3.shape))
    



    ## HERE NETWORK DEFINITION IS FINISHED
    
    ###  NOW CALCULATE PREDICTED VALUE
    #with tf.name_scope('calculate_predictions'):
    # output_shape = tf.shape(self.y_outputs)
    # self.predictions = tf.nn.softmax(tf.reshape(self.y_outputs, [-1, self.output_size]))
     
   ##
   ## CALCULATE LOSS
   ##
    with tf.name_scope('calculate_loss'):
     self.cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values_1,logits=self.y_outputs_1)
     self.loss_1 = tf.reduce_mean(self.cross_entropy_1)
     cross_entropy_2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values_2,logits=self.y_outputs_2)
     self.loss_2 = tf.reduce_mean(cross_entropy_2)
     cross_entropy_3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values_3,logits=self.y_outputs_3)
     self.loss_3 = tf.reduce_mean(cross_entropy_3)

     ## M.P. UYARI : BURADA MSE yerine SOFTMAX kullaninca hep sifir cikiyor. SIGMOID kullaninca oluyor http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
     # https://gombru.github.io/2018/05/23/cross_entropy_loss/
     self.cross_entropy_adverserial_1_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_y_values_adverserial_1_2,logits=self.y_outputs_adverserial_1_2)
     self.loss_adverserial_1_2 = tf.reduce_mean(self.cross_entropy_adverserial_1_2)
     self.cross_entropy_adverserial_1_3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_y_values_adverserial_1_3,logits=self.y_outputs_adverserial_1_3)
     self.loss_adverserial_1_3 = tf.reduce_mean(self.cross_entropy_adverserial_1_3)
     self.cross_entropy_adverserial_2_3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_y_values_adverserial_2_3,logits=self.y_outputs_adverserial_2_3)
     self.loss_adverserial_2_3 = tf.reduce_mean(self.cross_entropy_adverserial_2_3)
     #self.loss_adverserial = tf.losses.mean_squared_error(labels=self.real_y_values_adverserial,predictions=self.y_outputs_adverserial)



     global LOSS_WEIGHT_1,LOSS_WEIGHT_2,LOSS_WEIGHT_3,LOSS_WEIGHT_ADVER_1_2,LOSS_WEIGHT_ADVER_1_3,LOSS_WEIGHT_ADVER_2_3
     self.loss_single=LOSS_WEIGHT_1*self.loss_1+LOSS_WEIGHT_2*self.loss_2+LOSS_WEIGHT_3*self.loss_2+LOSS_WEIGHT_ADVER_1_2*self.loss_adverserial_1_2+LOSS_WEIGHT_ADVER_1_3*self.loss_adverserial_1_3+LOSS_WEIGHT_ADVER_2_3*self.loss_adverserial_2_3


  ## ADVERSERIAL FC iki ciktinin ayni olup olmadigini soyler 0/1
  ## gercekte ayni olup olmadigina gore duzeltme verilir.


   ##
   ## SET OPTIMIZER
   ##
   with tf.name_scope('optimizer'):
    self.optimizer_single = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.learning_rate_beta1,beta2=self.learning_rate_beta2).minimize(self.loss_single)

   ##
   ## CALCULATE ACCURACY
   ##
   with tf.name_scope('calculate_accuracy'):
    correct_prediction_1 = tf.equal(tf.argmax(self.y_outputs_1, 1),tf.argmax(self.real_y_values_1, 1))
    correct_prediction_1 = tf.cast(correct_prediction_1, tf.float32)
    self.accuracy_1 = tf.reduce_mean(correct_prediction_1)
    correct_prediction_2 = tf.equal(tf.argmax(self.y_outputs_2, 1),tf.argmax(self.real_y_values_2, 1))
    correct_prediction_2 = tf.cast(correct_prediction_2, tf.float32)
    self.accuracy_2 = tf.reduce_mean(correct_prediction_2)
    correct_prediction_3 = tf.equal(tf.argmax(self.y_outputs_3, 1),tf.argmax(self.real_y_values_3, 1))
    correct_prediction_3 = tf.cast(correct_prediction_3, tf.float32)
    self.accuracy_3 = tf.reduce_mean(correct_prediction_3)

    #self.correct_prediction_adverserial = tf.equal(tf.argmax(self.y_outputs_adverserial, 1),tf.argmax(self.real_y_values_adverserial, 1))
    #self.correct_prediction_adverserial = tf.cast(self.correct_prediction_adverserial, tf.float32)
    #self.accuracy_adverserial = tf.reduce_mean(self.correct_prediction_adverserial)
    #self.accuracy_adverserial  = tf.metrics.mean_squared_error(labels=self.real_y_values_adverserial, predictions=self.y_outputs_adverserial)

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

 def train(self,data1,data2,data3):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data1,y_data1=self.prepareData(data1,augment)
  x_data2,y_data2=self.prepareData(data2,augment)
  x_data3,y_data3=self.prepareData(data3,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 


  y_values_adverserial_1_2=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data1[i], y_data2[i]) :
          y_values_adverserial_1_2[i][0]=1

  y_values_adverserial_1_3=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data1[i], y_data3[i]) :
          y_values_adverserial_1_3[i][0]=1

  y_values_adverserial_2_3=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data2[i], y_data3[i]) :
          y_values_adverserial_2_3[i][0]=1

  self.optimizer_single.run(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob:self.keep_prob_constant})

  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingAccuracy_1 = self.accuracy_1.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  trainingAccuracy_2 = self.accuracy_2.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  trainingAccuracy_3 = self.accuracy_3.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  y_outputs_adverserial_1_2 = self.y_outputs_adverserial_1_2.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  y_outputs_adverserial_1_3 = self.y_outputs_adverserial_1_3.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  y_outputs_adverserial_2_3 = self.y_outputs_adverserial_2_3.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})

  loss_single = self.loss_single.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  
  trainingAccuracy_adverserial_1_2=0
  for i in range(y_outputs_adverserial_1_2.shape[0]):
    if (y_values_adverserial_1_2[i] == 0 and y_outputs_adverserial_1_2[i] < self.ADVERSERIAL_TRESHOLD) or  (y_values_adverserial_1_2[i] == 1 and y_outputs_adverserial_1_2[i] > self.ADVERSERIAL_TRESHOLD):
       trainingAccuracy_adverserial_1_2=trainingAccuracy_adverserial_1_2+1
  trainingAccuracy_adverserial_1_2= trainingAccuracy_adverserial_1_2/y_outputs_adverserial_1_2.shape[0]    
  
  trainingAccuracy_adverserial_1_3=0
  for i in range(y_outputs_adverserial_1_3.shape[0]):
    if (y_values_adverserial_1_3[i] == 0 and y_outputs_adverserial_1_3[i] < self.ADVERSERIAL_TRESHOLD) or  (y_values_adverserial_1_3[i] == 1 and y_outputs_adverserial_1_3[i] > self.ADVERSERIAL_TRESHOLD):
       trainingAccuracy_adverserial_1_3=trainingAccuracy_adverserial_1_3+1
  trainingAccuracy_adverserial_1_3= trainingAccuracy_adverserial_1_3/y_outputs_adverserial_1_3.shape[0]    
  
  trainingAccuracy_adverserial_2_3=0
  for i in range(y_outputs_adverserial_2_3.shape[0]):
    if (y_values_adverserial_2_3[i] == 0 and y_outputs_adverserial_2_3[i] < self.ADVERSERIAL_TRESHOLD) or  (y_values_adverserial_2_3[i] == 1 and y_outputs_adverserial_2_3[i] > self.ADVERSERIAL_TRESHOLD):
       trainingAccuracy_adverserial_2_3=trainingAccuracy_adverserial_2_3+1
  trainingAccuracy_adverserial_2_3= trainingAccuracy_adverserial_2_3/y_outputs_adverserial_2_3.shape[0]    
  

#  cross_entropy_1 = self.cross_entropy_1.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2 ,self.real_y_values_adverserial:y_values_adverserial,self.keep_prob: 1.0})
#  cross_entropy_adverserial = self.cross_entropy_adverserial.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2 ,self.real_y_values_adverserial:y_values_adverserial,self.keep_prob: 1.0})

#  print("---------------------------------------------")
#  print("loss_1:")
#  print(loss_1)
#  print("loss_2:")
#  print(loss_2)
#  print("loss_adverserial:")
#  print(loss_adverserial)
#  print("trainingAccuracy_adverserial:")
#  print(trainingAccuracy_adverserial)
#  print("y_outputs_adverserial:")
#  print(*y_outputs_adverserial)
#  print("real values:")
#  print(*y_values_adverserial)
#  print("")
#  print("cross_entropy_1:")
#  print(*cross_entropy_1)
#  print("cross_entropy_adverserial:")
#  print(*cross_entropy_adverserial)
#  print("")

  #return trainingTime,trainingAccuracy_1,trainingAccuracy_2,trainingAccuracy_adverserial,loss_adverserial,prepareDataTime
  return trainingTime,trainingAccuracy_1,trainingAccuracy_2,trainingAccuracy_3,trainingAccuracy_adverserial_1_2,trainingAccuracy_adverserial_1_3,trainingAccuracy_adverserial_2_3,loss_single,prepareDataTime
     
 def test(self,data_1,data_2,data_3):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data1,y_data1=self.prepareData(data_1,augment) 
  x_data2,y_data2=self.prepareData(data_2,augment) 
  x_data3,y_data3=self.prepareData(data_3,augment) 

  y_values_adverserial_1_2=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data1[i], y_data2[i]) :
          y_values_adverserial_1_2[i][0]=1

  y_values_adverserial_1_3=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data1[i], y_data3[i]) :
          y_values_adverserial_1_3[i][0]=1

  y_values_adverserial_2_3=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data2[i], y_data3[i]) :
          y_values_adverserial_2_3[i][0]=1

  y_outputs_adverserial_1_2 = self.y_outputs_adverserial_1_2.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  y_outputs_adverserial_1_3 = self.y_outputs_adverserial_1_3.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})
  y_outputs_adverserial_2_3 = self.y_outputs_adverserial_2_3.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3,self.keep_prob: 1.0})

  testAccuracy_1 = self.accuracy_1.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3, self.keep_prob: 1.0})
  testAccuracy_2 = self.accuracy_2.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3, self.keep_prob: 1.0})
  testAccuracy_3 = self.accuracy_3.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.x_input_3: x_data3, self.real_y_values_3:y_data3,self.real_y_values_adverserial_1_2:y_values_adverserial_1_2,self.real_y_values_adverserial_1_3:y_values_adverserial_1_3,self.real_y_values_adverserial_2_3:y_values_adverserial_2_3, self.keep_prob: 1.0})
 
  testAccuracy_adverserial_1_2=0
  for i in range(y_outputs_adverserial_1_2.shape[0]):
    if (y_values_adverserial_1_2[i] == 0 and y_outputs_adverserial_1_2[i] < self.ADVERSERIAL_TRESHOLD) or  (y_values_adverserial_1_2[i] == 1 and y_outputs_adverserial_1_2[i] > self.ADVERSERIAL_TRESHOLD):
       testAccuracy_adverserial_1_2=testAccuracy_adverserial_1_2+1
  testAccuracy_adverserial_1_2= testAccuracy_adverserial_1_2/y_outputs_adverserial_1_2.shape[0]    
  
  testAccuracy_adverserial_1_3=0
  for i in range(y_outputs_adverserial_1_3.shape[0]):
    if (y_values_adverserial_1_3[i] == 0 and y_outputs_adverserial_1_3[i] < self.ADVERSERIAL_TRESHOLD) or  (y_values_adverserial_1_3[i] == 1 and y_outputs_adverserial_1_3[i] > self.ADVERSERIAL_TRESHOLD):
       testAccuracy_adverserial_1_3=testAccuracy_adverserial_1_3+1
  testAccuracy_adverserial_1_3= testAccuracy_adverserial_1_3/y_outputs_adverserial_1_3.shape[0]    
  
  testAccuracy_adverserial_2_3=0
  for i in range(y_outputs_adverserial_2_3.shape[0]):
    if (y_values_adverserial_2_3[i] == 0 and y_outputs_adverserial_2_3[i] < self.ADVERSERIAL_TRESHOLD) or  (y_values_adverserial_2_3[i] == 1 and y_outputs_adverserial_2_3[i] > self.ADVERSERIAL_TRESHOLD):
       testAccuracy_adverserial_2_3=testAccuracy_adverserial_2_3+1
  testAccuracy_adverserial_2_3= testAccuracy_adverserial_2_3/y_outputs_adverserial_2_3.shape[0]    
  
 
  #testAccuracy = self.accuracy.eval(feed_dict={self.x_input_1: x_data, self.real_y_values_1:y_data,self.x_input_2: x_data, self.real_y_values_2:y_data, self.keep_prob: 1.0})
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy_1,testAccuracy_2,testAccuracy_3,testAccuracy_adverserial_1_2,testAccuracy_adverserial_1_3,testAccuracy_adverserial_2_3
  

