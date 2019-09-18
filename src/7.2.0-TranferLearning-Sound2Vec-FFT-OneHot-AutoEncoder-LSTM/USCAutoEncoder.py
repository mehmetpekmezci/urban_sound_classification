#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *





##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class USCAutoEncoder :
 def __init__(self, session, uscLogger,uscData): 
   self.session               = session
   self.uscLogger             = uscLogger
   self.uscData               = uscData
   self.time_window_size      = 10
   script_dir=os.path.dirname(os.path.realpath(__file__))
   script_name=os.path.basename(script_dir)
   self.model_save_dir=script_dir+"/../../save/"+script_name
   self.model_save_file="autoEndoer.h5"
   ## self.uscData.time_slice_length = 440
   ## so we will have nearly 400 time steps in 4 secs record. (88200) (with %50 overlapping)
   ## so we again sliced the input data into 20 (num_of_paral_lstms)
   ## each lstm cell works on one part only (for example lstm[0] works on the begginning of the data
   ## (self.num_of_paralel_lstms  = 28) x (self.lstm_time_steps = 18) = (self.uscData.number_of_time_slices=504)
   self.training_iterations   = 10
   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True
   self.model,self.encoder=self.buildModel()
   self.load_weights()
   self.trainCount=0

 def load_weights(self):
     if os.path.exists(self.model_save_dir+"/"+self.model_save_file):
         self.model.load_weights(self.model_save_dir+"/"+self.model_save_file)

 def save_weights(self):
     if not os.path.exists(self.model_save_dir):
         os.makedirs(self.model_save_dir)
     self.model.save_weights(self.model_save_dir+"/"+self.model_save_file)


 def prepareData(self,data,augment):
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  x_data=self.uscData.overlapping_slice(x_data)
  ## returns -> (batch_size, number_of_time_slices, time_slice_length)
  x_data=self.uscData.fft(x_data)
  x_data=self.uscData.peaksMultiHot(x_data)
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
   y_data=y_data.reshape(y_data.shape[0],y_data.shape[1]*y_data.shape[2])
   x_data=x_data[:,int(x_data.shape[1]/2),:].reshape(x_data.shape[0],1,x_data.shape[2])
   self.model.fit(x_data, y_data, epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingLossTotal=0

  test_data_x=None
  test_data_y=None
  for x_data in x_data_list :
   y_data=np.delete(x_data,int(x_data.shape[1]/2),1)
   y_data=y_data.reshape(y_data.shape[0],y_data.shape[1]*y_data.shape[2])
   x_data=x_data[:,int(x_data.shape[1]/2),:].reshape(x_data.shape[0],1,x_data.shape[2])
   test_data_x=x_data
   test_data_y=y_data
   evaluation = self.model.evaluate(x_data, y_data, batch_size = self.uscData.mini_batch_size,verbose=0)
   #print(self.model.metrics_names)
   #print(evaluation)
   trainingLossTotal+=evaluation[0]
  #print(trainingLossTotal)
  trainingLoss=trainingLossTotal/len(x_data_list)


  self.uscLogger.logger.info("x_data.shape : "+str(test_data_x.shape))
  self.uscLogger.logger.info("y_data.shape : "+str(test_data_y.shape))
  self.uscLogger.logger.info("x_data[0].shape : "+str(test_data_x[0].shape))
  self.uscLogger.logger.info("y_data[0].shape : "+str(test_data_y[0].shape))
  test_data_x=test_data_x[0].reshape(1,1,test_data_x[0].shape[1])
  test_data_y=test_data_y[0].reshape(1,test_data_y[0].shape[0])
  self.uscLogger.logger.info("x_data.shape : "+str(test_data_x.shape))
  self.uscLogger.logger.info("y_data.shape : "+str(test_data_y.shape))
  self.uscLogger.logger.info("x_data : "+str(test_data_x))
  self.uscLogger.logger.info("y_data : "+str(test_data_y))
  self.uscLogger.logger.info("predicted_data : "+str(self.model.predict(test_data_x)))

  #print(self.model.metrics_names) 
  #print(evaluation) 
  self.trainCount+=1
  
  if self.trainCount % 100 :
     self.save_weights()
  
  return trainingTime,trainingLoss,prepareDataTime
     

 def buildModel(self):
   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,1,self.uscData.time_slice_length))
#   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.word2vec_window_size,self.uscData.time_slice_length))
   x = keras.layers.Convolution1D(16, 3,activation='relu', border_mode='same')(layer_input) #nb_filter, nb_row, nb_col
   x = keras.layers.MaxPooling1D((5), border_mode='same')(x)
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.MaxPooling1D((5), border_mode='same')(x)
   x = keras.layers.Convolution1D(4, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.Convolution1D(4, 3, activation='relu', border_mode='same')(x)

   x = keras.layers.Convolution1D(2, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.Convolution1D(2, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.Convolution1D(1, 3, activation='relu', border_mode='same')(x)

   encoded = x  # (self.uscData.mini_batch_size,self.uscData.latent_space_presentation_data_length)
   #encoded = keras.layers.MaxPooling1D((5), border_mode='same')(x)  # (self.uscData.mini_batch_size,self.uscData.latent_space_presentation_data_length)
   self.uscLogger.logger.info("shape of encoder"+str(encoded.shape))
   self.uscData.latent_space_presentation_data_length=int(encoded.shape[1])
   
   x = keras.layers.Convolution1D(1, 3, activation='relu', border_mode='same')(encoded)
   x = keras.layers.Convolution1D(2, 3, activation='relu', border_mode='same')(encoded)
   x = keras.layers.Convolution1D(2, 3, activation='relu', border_mode='same')(encoded)

   x = keras.layers.Convolution1D(4, 3, activation='relu', border_mode='same')(encoded)
   x = keras.layers.Convolution1D(4, 3, activation='relu', border_mode='same')(encoded)
   x = keras.layers.UpSampling1D((5))(x)
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.UpSampling1D((5))(x)
   x = keras.layers.Convolution1D(16, 3, activation='relu', border_mode='same')(x) 
   #x = keras.layers.UpSampling1D((int(3*(self.uscData.word2vec_window_size-1))))(x)
   
   ##x = keras.layers.Convolution1D(1,3, activation='sigmoid', border_mode='same')(x)
   x =  keras.layers.Flatten()(x)
   decoded =keras.layers.Dense(units =(self.uscData.word2vec_window_size-1)*self.uscData.time_slice_length,activation='softmax')(x)

   self.uscLogger.logger.info("shape of decoded "+str( decoded.shape))

   autoencoder = keras.models.Model(layer_input,decoded)
   flattennedEncoded=keras.layers.Flatten()(encoded)
   encoder = keras.models.Model(layer_input,flattennedEncoded)
   selectedOptimizer=keras.optimizers.Adam(lr=0.001)
   #autoencoder.compile(optimizer=selectedOptimizer, loss='binary_crossentropy')
   autoencoder.compile(optimizer=selectedOptimizer, loss='categorical_crossentropy',metrics=['accuracy'])
   #autoencoder.compile(optimizer=selectedOptimizer, loss='mse')
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
       x_data_item=x_data_item.reshape(x_data_item.shape[0],1,x_data_item.shape[1])
       encoded_x_data_item=self.encoder.predict(x_data_item)
       #self.uscLogger.logger.info("encoded_x_data_item.shape="+str( encoded_x_data_item.shape))
       x_encoded_data_list.append(encoded_x_data_item)
   encoded_x_data=np.asarray(x_encoded_data_list)
   encodedValue=np.swapaxes(encoded_x_data,0,1)
   #self.uscLogger.logger.info("encodedValue.shape="+str( encodedValue.shape))
   encodeTimeStop = int(round(time.time()))
   encodeTime=encodeTimeStop-encodeTimeStart
   return encodedValue,encodeTime    






