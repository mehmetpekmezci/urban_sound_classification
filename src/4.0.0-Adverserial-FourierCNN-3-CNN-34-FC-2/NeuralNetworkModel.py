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
              fourier_cnn_layers=FOURIER_CNN_LAYERS):

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
   self.fourier_cnn_layers=fourier_cnn_layers


   self.keep_prob = tf.placeholder(tf.float32)

   

   ##
   ## DEFINE PLACE HOLDER FOR REAL OUTPUT VALUES FOR TRAINING
   ##
   self.real_y_values_1=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values_1")
   self.real_y_values_2=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values_2")
   self.real_y_values_adverserial=tf.placeholder(tf.float32, shape=(self.mini_batch_size, 1), name="real_y_values_adverserial")

   ##
   ## BUILD THE NETWORK
   ##

   ##
   ## INPUT  LAYER
   ##
   number_of_input_channels=1
   self.x_input_1                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input_1")
   self.x_input_2                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input_2")
   last_layer_output_1=self.x_input_1
   last_layer_output_2=self.x_input_2

   ##
   ## RESHAPE
   ##
   with tf.name_scope('input_reshape'):
     self.x_input_reshaped_1 = tf.reshape(last_layer_output_1, [self.mini_batch_size, self.input_size_y, int(last_layer_output_1.shape[1]), number_of_input_channels])
     self.x_input_reshaped_2 = tf.reshape(last_layer_output_2, [self.mini_batch_size, self.input_size_y, int(last_layer_output_2.shape[1]), number_of_input_channels])
     self.logger.info("self.x_input_reshaped_1.shape="+str(self.x_input_reshaped_1.shape))
     self.logger.info("self.x_input_reshaped_2.shape="+str(self.x_input_reshaped_2.shape))
     previous_level_convolution_output_1 = self.x_input_reshaped_1
     previous_level_convolution_output_2 = self.x_input_reshaped_2


   ##
   ## FOURIER  CNN LAYERS
   ##
   with tf.name_scope('fourier_CNN_1'):
    for fourierCNNLayerNo in range(len(self.fourier_cnn_layers)) :
     self.logger.info("previous_level_convolution_output_1.shape="+str(previous_level_convolution_output_1.shape))
     cnnLayerName    = "fourier-cnn-"+str(fourierCNNLayerNo)     
     cnnKernelCount  = self.fourier_cnn_layers[fourierCNNLayerNo]   
     # cnnKernelCount tane cnnKernelSizeX * cnnKernelSizeY lik convolution kernel uygulanacak , sonucta 64x1x88200 luk tensor cikacak.
     cnnKernelSizeX  = 1
     cnnKernelSizeY  = 3        
     cnnStrideSizeX  = 1 
     cnnStrideSizeY  = 1                     
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
       C = tf.nn.conv2d(previous_level_convolution_output_1,W,strides=[1,cnnStrideSizeX, cnnStrideSizeY, 1], padding='SAME')+B

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
       previous_level_convolution_output_1=P
       self.logger.info(cnnLayerName+".H_pooled.shape="+str(P.shape))
     else :
       ## no residual for layer liner CNN as fourier transform.
       previous_level_convolution_output_1=H

     previous_level_kernel_count=cnnKernelCount
     fourierCNNOutput_1=previous_level_convolution_output_1


   ##
   ## FOURIER  CNN LAYERS
   ##
   with tf.name_scope('fourier_CNN_2'):
    for fourierCNNLayerNo in range(len(self.fourier_cnn_layers)) :
     self.logger.info("previous_level_convolution_output_2.shape="+str(previous_level_convolution_output_2.shape))
     cnnLayerName    = "fourier-cnn-"+str(fourierCNNLayerNo)     
     cnnKernelCount  = self.fourier_cnn_layers[fourierCNNLayerNo]   
     # cnnKernelCount tane cnnKernelSizeX * cnnKernelSizeY lik convolution kernel uygulanacak , sonucta 64x1x88200 luk tensor cikacak.
     cnnKernelSizeX  = 1
     cnnKernelSizeY  = 3        
     cnnStrideSizeX  = 1 
     cnnStrideSizeY  = 1                     
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
       C = tf.nn.conv2d(previous_level_convolution_output_2,W,strides=[1,cnnStrideSizeX, cnnStrideSizeY, 1], padding='SAME')+B

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
       previous_level_convolution_output_2=P
       self.logger.info(cnnLayerName+".H_pooled.shape="+str(P.shape))
     else :
       ## no residual for layer liner CNN as fourier transform.
       previous_level_convolution_output_2=H

     previous_level_kernel_count=cnnKernelCount
     fourierCNNOutput_2=previous_level_convolution_output_2


    previous_level_convolution_output=tf.concat((previous_level_convolution_output_1,previous_level_convolution_output_2),1)


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


   ##
   ## DOUBLE OUTPUT
   ##


   ## FISRT FULLY CONNECTED 
   last_layer_output=cnn_last_layer_output_flat
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
     first_fc_output=last_layer_output


   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs_1 =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs_1.shape="+str(self.y_outputs_1.shape))
      


   ## SECOND FULLY CONNECTED 

   last_layer_output=cnn_last_layer_output_flat
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
     second_fc_output=last_layer_output

   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, self.output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[self.output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs_2 =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs_2.shape="+str(self.y_outputs_2.shape))
    

   # first_fc_output ve second_fc_output birlestir
   # Adverserial fully connected layer a ver
   last_layer_output=tf.concat((first_fc_output,second_fc_output),1)
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

   #adverserial output=0/1  yes or no, meaning that these two outputs are the same or not
   adverserial_output_size=1
   # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS features to OUTPUT_SIZE=NUMBER_OF_CLASSES(10) classes, one for each class
   with tf.name_scope('last_fc'):
    W_fc2 =  tf.Variable( tf.truncated_normal([number_of_fully_connected_layer_neurons, adverserial_output_size], stddev=0.1))
    b_fc2 =  tf.Variable(tf.constant(0.1, shape=[adverserial_output_size]))
    #h_fc2 =tf.nn.relu( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    self.y_outputs_adverserial =tf.matmul(last_layer_output, W_fc2) + b_fc2
    self.logger.info("self.y_outputs_adverserial.shape="+str(self.y_outputs_adverserial.shape))
    
    ## HERE NETWORK DEFINITION IS FINISHED
    
    ###  NOW CALCULATE PREDICTED VALUE
    #with tf.name_scope('calculate_predictions'):
    # output_shape = tf.shape(self.y_outputs)
    # self.predictions = tf.nn.softmax(tf.reshape(self.y_outputs, [-1, self.output_size]))
     
   ##
   ## CALCULATE LOSS
   ##
    with tf.name_scope('calculate_loss'):
     cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values_1,logits=self.y_outputs_1)
     self.loss_1 = tf.reduce_mean(cross_entropy_1)
     cross_entropy_2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values_2,logits=self.y_outputs_2)
     self.loss_2 = tf.reduce_mean(cross_entropy_2)
     cross_entropy_adverserial = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_y_values_adverserial,logits=self.y_outputs_adverserial)
     self.loss_adverserial = tf.reduce_mean(cross_entropy_adverserial)

  ## ADVERSERIAL FC iki ciktinin ayni olup olmadigini soyler 0/1
  ## gercekte ayni oluğ olmadigina gore duzeltme verilir.


   ##
   ## SET OPTIMIZER
   ##
   with tf.name_scope('optimizer'):
    self.optimizer_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.learning_rate_beta1,beta2=self.learning_rate_beta2).minimize(self.loss_1)
    self.optimizer_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.learning_rate_beta1,beta2=self.learning_rate_beta2).minimize(self.loss_2)
    self.optimizer_adverserial = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.learning_rate_beta1,beta2=self.learning_rate_beta2).minimize(self.loss_adverserial)

   ##
   ## CALCULATE ACCURACY
   ##
   with tf.name_scope('calculate_accuracy'):
    correct_prediction = tf.equal(tf.argmax(self.y_outputs_1, 1),tf.argmax(self.real_y_values_1, 1))
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

 def train(self,data1,data2):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data1,y_data1=self.prepareData(data1,augment)
  x_data2,y_data2=self.prepareData(data2,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 


  y_values_adverserial=np.zeros((self.mini_batch_size,1)) # [0,0,0,0,0]
  for i in range(self.mini_batch_size) :
    if  np.array_equal(y_data1[i], y_data2[i]) :
          y_values_adverserial[i][0]=1


  self.optimizer_1.run(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.real_y_values_adverserial:y_values_adverserial,self.keep_prob:self.keep_prob_constant})
  self.optimizer_2.run(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.real_y_values_adverserial:y_values_adverserial,self.keep_prob:self.keep_prob_constant})
  self.optimizer_adverserial.run(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2,self.real_y_values_adverserial:y_values_adverserial,self.keep_prob:self.keep_prob_constant})

  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingAccuracy = self.accuracy.eval(feed_dict={self.x_input_1: x_data1, self.real_y_values_1:y_data1,self.x_input_2: x_data2, self.real_y_values_2:y_data2, self.keep_prob: 1.0})
  return trainingTime,trainingAccuracy,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data,y_data=self.prepareData(data,augment) 
  testAccuracy = self.accuracy.eval(feed_dict={self.x_input_1: x_data, self.real_y_values_1:y_data,self.x_input_2: x_data, self.real_y_values_2:y_data, self.keep_prob: 1.0})
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  



