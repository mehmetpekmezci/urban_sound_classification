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
   script_name=os.path.basename(self.script_dir)
   self.model_save_file=script_dir+"/../../save/"+script_name+"/autoEndoer.h5")
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
     if os.path.exists(self.model_save_file):
         self. model.load_weights(self.model_save_file)

 def save_weights(self):
     if os.path.exists(self.model_save_file):
          self.model.save_weights(self.model_save_file)

 def prepareData(self,data,augment):
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.overlapping_slice(x_data,hanning=False)
  ## returns -> (batch_size, number_of_time_slices, time_slice_length)
  #x_data=self.uscData.fft(x_data)
  x_data=self.uscData.normalize(x_data)
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
   y_data=y_data.rehape(y_data.shape[0],y_data.shape[1]*y_data.shape[2])
   print(y_data.shape)
   x_data=x_data[:,int(x_data.shape[1]/2),:].rehape(x_data.shape[0],x_data.shape[2])
   print(x_data.shape)
   self.model.fit(x_data, y_data, epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingAccuracyTotal=0
  for x_data in x_data_list :
   evaluation = self.model.evaluate(x_data, y_data, batch_size = self.uscData.mini_batch_size,verbose=0)
   trainingAccuracyTotal+=evaluation[1]
  trainingAccuracy=trainingAccuracyTotal/len(x_data_list)
  #print(self.model.metrics_names) 
  #print(evaluation) 
  self.trainCount+=1
  
  if self.trainCount % 100 :
     self.save_weights()
  
  return trainingTime,trainingAccuracy,prepareDataTime
     

 def buildModel(self):
   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.time_slice_length))
#   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.word2vec_window_size,self.uscData.time_slice_length))
   x = keras.layers.Convolution1D(16, 3,activation='relu', border_mode='same')(layer_input) #nb_filter, nb_row, nb_col
   x = keras.layers.MaxPooling1D((2), border_mode='same')(x)
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   x = keras.layers.MaxPooling1D((2), border_mode='same')(x)
   x = keras.layers.Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   self.encoder = keras.layers.MaxPooling1D((2), border_mode='same')(x)  # (self.uscData.mini_batch_size,self.uscData.latent_space_presentation_data_length)
   #print ("shape of self.encoder ", K.int_shape(self.encoder ))
   self.uscLogger.info("shape of encoder"+str( self.encoder.shape))
   x = Convolution1D(8, 3, activation='relu', border_mode='same')(self.encoder)
   x = UpSampling1D((2))(x)
   x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
   x = UpSampling1D((2))(x)

   # In original tutorial, border_mode='same' was used. 
   # then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
   # x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) 
   x = Convolution1D(16, 3, activation='relu', border_mode='valid')(x) 

   x = UpSampling2D((2))(x)
   decoded = Convolution1D(int(self.uscData.word2vec_window_size-1),1, activation='sigmoid', border_mode='same')(x)
   self.uscLogger.info("shape of decoded"+str( decoded.shape))

   autoencoder = Model(layer_input, decoded)
   selectedOptimizer=keras.optimizers.Adam(lr=0.0001)
   #autoencoder.compile(optimizer=selectedOptimizer, loss='binary_crossentropy')
   #autoencoder.compile(optimizer=selectedOptimizer, loss='categorical_crossentropy',metrics=['accuracy'])
   autoencoder.compile(optimizer=selectedOptimizer, loss='mse')
   return model

#   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)
#   model = keras.models.Model(inputs=inputs, outputs=[out])

 def encode(self,x_data):
   encodeTimeStart = int(round(time.time()))
   x_data_list=np.swapaxes(x_data,0,1).to_list() 
   x_encoded_data_list=[]
   ##  (self.mini_batch_size      ,self.number_of_time_slices,self.time_slice_length)  to
   ##  (self.number_of_time_slices,self.mini_batch_size      ,self.time_slice_length)
   for x_data in x_data_list :
       x_encoded_data_list.append(self.encoder.predict(x_data))
   encoded_x_data=np.asarray(x_encoded_data_list)
   encodedValue=np.swapaxes(encoded_x_data,0,1)
   self.uscLogger.info("encodedValue.shape="+str( encodedValue.shape))
   encodeTimeStop = int(round(time.time()))
   encodeTime=encodeTimeStop-encodeTimeStart
   return encodedValue,encodeTime    






