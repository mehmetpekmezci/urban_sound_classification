#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class AutoEncoder :
 def __init__(self,session,logger,input_size=INPUT_SIZE,learning_rate=LEARNING_RATE,
              cnn_kernel_counts=AE_CNN_KERNEL_COUNTS,
              cnn_kernel_x_sizes=AE_CNN_KERNEL_X_SIZES,cnn_kernel_y_sizes=AE_CNN_KERNEL_Y_SIZES,
              cnn_stride_x_sizes=AE_CNN_STRIDE_X_SIZES,cnn_stride_y_sizes=AE_CNN_STRIDE_Y_SIZES,
              cnn_pool_x_sizes=AE_CNN_POOL_X_SIZES,cnn_pool_y_sizes=AE_CNN_POOL_Y_SIZES,
              mini_batch_size=int(MINI_BATCH_SIZE+MINI_BATCH_SIZE_FOR_GENERATED_DATA),
              encoder_layers=ENCODER_LAYERS,keep_prob_constant=KEEP_PROB,epsilon=EPSILON):

   ##
   ## SET CLASS ATTRIBUTES WITH THE GIVEN INPUTS
   ##
   self.session               = session
   self.logger                = logger
   self.input_size            = input_size
   self.input_size_y          = 1
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
   self.x_input                      = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")



   ##
   ## FOURIER  CNN LAYERS
   ##
   with tf.name_scope('fourier_CNN'):
    for fourierCNNLayerNo in range(3) :
     self.logger.info("auto encoder fourier layers previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "ae fourier-cnn-"+str(fourierCNNLayerNo)     
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
     fourierCNNOutput=previous_level_convolution_output





   ##
   ## CNN LAYERS
   ##

    for cnnLayerNo in range(len(self.cnn_kernel_counts)) :
     self.logger.info("previous_level_convolution_output.shape="+str(previous_level_convolution_output.shape))
     cnnLayerName    = "cnn-"+str(cnnLayerNo)     


----------------------------

   last_layer_output=self.x_input
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
   self.encoder=last_layer_output
   
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
     self.loss=tf.losses.mean_squared_error(self.x_input,self.y_output)

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

 def prepareData(self,data,generated_data):
  x_data=augment_random(data[:,:4*SOUND_RECORD_SAMPLING_RATE])
  concat_data=np.concatenate((x_data,generated_data),axis=0)
  x_data=np.random.permutation(concat_data)
  return x_data

 def train(self,data,generated_data):
  prepareDataTimeStart = int(round(time.time())) 
  x_data=self.prepareData(data,generated_data)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  self.optimizer.run(feed_dict={self.x_input: x_data,self.keep_prob: self.keep_prob_constant})
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingLoss = self.loss.eval(feed_dict={self.x_input: x_data,self.keep_prob: 1.0})
  return trainingTime,trainingLoss,prepareDataTime
     
 def encode(self,data):
  encodeTimeStart = int(round(time.time())) 
  x_data=data[:,:4*SOUND_RECORD_SAMPLING_RATE]
  encodedValue = self.encoder.eval(feed_dict={self.x_input: x_data, self.keep_prob: 1.0})
  encodeTimeStop = int(round(time.time())) 
  encodeTime=encodeTimeStop-encodeTimeStart
  return encodedValue,encodeTime
  
  
  
  
--------------------------------------------  
  
  
  
   def prepareData(self,data,augment):
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  x_data=self.uscData.overlapping_slice(x_data)
  ## returns -> (batch_size, number_of_time_slices, time_slice_length)
  #x_data=self.uscData.fft(x_data)
  x_data_list = self.uscData.convert_to_list_of_word2vec_window_sized_data(x_data)
  ## returns -> list of (mini_batch_size,word2vec_window_size,time_slice_lentgh), this list has self.number_of_time_slices/self.word2vec_window_size elements
  return x_data_list


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data_list=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  for x_data in x_data_list :
   ## x_data of size (mini_batch_size,word2vec_window_size,time_slice_lentgh)
  ## pull the data in the middle
   #y_data=np.concatenate((x_data[:,:int(len(x_data)/2+1),:],x_data[:,int(len(x_data)/2+1):,:]))
   y_data=np.delete(x_data,int(x_data.shape[1]/2),1)
   y_data=y_data.reshape(y_data.shape[0],y_data.shape[1]*y_data.shape[2],1)
   x_data=x_data[:,int(x_data.shape[1]/2),:].reshape(x_data.shape[0],x_data.shape[2],1)
   self.model.fit(x_data, y_data, epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingLossTotal=0
  for x_data in x_data_list :
   y_data=np.delete(x_data,int(x_data.shape[1]/2),1)
   y_data=y_data.reshape(y_data.shape[0],y_data.shape[1]*y_data.shape[2],1)
   x_data=x_data[:,int(x_data.shape[1]/2),:].reshape(x_data.shape[0],x_data.shape[2],1)
   evaluation = self.model.evaluate(x_data, y_data, batch_size = self.uscData.mini_batch_size,verbose=0)
   #trainingLossTotal+=evaluation[0]
   trainingLossTotal+=evaluation
  trainingLoss=trainingLossTotal/len(x_data_list)
  #print(self.model.metrics_names) 
  #print(evaluation) 
  self.trainCount+=1
  
  if self.trainCount % 100 :
     self.save_weights()
  
  return trainingTime,trainingLoss,prepareDataTime
     

 def buildModel(self):
   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.time_slice_length,1))
#   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.word2vec_window_size,self.uscData.time_slice_length))
   x = keras.layers.Convolution1D(16, 3,activation='relu', border_mode='same')(layer_input) #nb_filter, nb_row, nb_col
   x = keras.layers.MaxPooling1D((5), border_mode='same')(x)
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.MaxPooling1D((5), border_mode='same')(x)
   x = keras.layers.Convolution1D(1, 3, activation='relu', border_mode='same')(x)
   encoded = keras.layers.MaxPooling1D((5), border_mode='same')(x)  # (self.uscData.mini_batch_size,self.uscData.latent_space_presentation_data_length)
   self.uscLogger.logger.info("shape of encoder"+str(encoded.shape))
   self.uscData.latent_space_presentation_data_length=int(encoded.shape[1])
   
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(encoded)
   x = keras.layers.UpSampling1D((5))(x)
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.UpSampling1D((5))(x)

   # In original tutorial, border_mode='same' was used. 
   # then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
   # x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) 
   x = keras.layers.Convolution1D(16, 3, activation='relu', border_mode='same')(x) 
   x = keras.layers.UpSampling1D((int(5*(self.uscData.word2vec_window_size-1))))(x)
   
   decoded = keras.layers.Convolution1D(1,3, activation='sigmoid', border_mode='same')(x)
   self.uscLogger.logger.info("shape of decoded "+str( decoded.shape))

   autoencoder = keras.models.Model(layer_input,decoded)
   flattennedEncoded=keras.layers.Flatten()(encoded)
   encoder = keras.models.Model(layer_input,flattennedEncoded)
   selectedOptimizer=keras.optimizers.Adam(lr=0.0001)
   #autoencoder.compile(optimizer=selectedOptimizer, loss='binary_crossentropy')
   #autoencoder.compile(optimizer=selectedOptimizer, loss='categorical_crossentropy',metrics=['accuracy'])
   autoencoder.compile(optimizer=selectedOptimizer, loss='mse')
   return autoencoder,encoder

#   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)
#   model = keras.models.Model(inputs=inputs, outputs=[out])

 def encode(self,x_data):
   encodeTimeStart = int(round(time.time()))
   x_data_swapped=np.swapaxes(x_data,0,1)
   x_data_list=[]
   for i in range(x_data_swapped.shape[0]):
       x_data_list.append(x_data_swapped[i])
   x_encoded_data_list=[]
   ##  (self.mini_batch_size      ,self.number_of_time_slices,self.time_slice_length)  to
   ##  (self.number_of_time_slices,self.mini_batch_size      ,self.time_slice_length)
   for x_data_item in x_data_list :
       #self.uscLogger.logger.info("len(x_data_item)="+str( len(x_data_item)))
       x_data_item=x_data_item.reshape(x_data_item.shape[0],x_data_item.shape[1],1)
       encoded_x_data_item=self.encoder.predict(x_data_item)
       #self.uscLogger.logger.info("encoded_x_data_item.shape="+str( encoded_x_data_item.shape))
       x_encoded_data_list.append(encoded_x_data_item)
   encoded_x_data=np.asarray(x_encoded_data_list)
   encodedValue=np.swapaxes(encoded_x_data,0,1)
   #self.uscLogger.logger.info("encodedValue.shape="+str( encodedValue.shape))
   encodeTimeStop = int(round(time.time()))
   encodeTime=encodeTimeStop-encodeTimeStart
   return encodedValue,encodeTime    


