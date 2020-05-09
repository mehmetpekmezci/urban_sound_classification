#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *

##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class USCModel :
 def __init__(self, uscLogger,uscData): 
   self.uscLogger             = uscLogger
   self.uscData               = uscData
   ## self.uscData.time_slice_length = 440

   script_dir=os.path.dirname(os.path.realpath(__file__))
   script_name=os.path.basename(script_dir)
   self.model_save_dir=script_dir+"/../../save/"+script_name
   self.model_save_file="model.h5"

   ## so we will have nearly 400 time steps in 4 secs record. (88200) (with %50 overlapping)
   self.lstm_time_steps       = self.uscData.number_of_time_slices
   self.training_iterations   = 20000
   self.buildModel()

   self.load_weights()
   self.trainCount=0

   self.model.summary(print_fn=uscLogger.logger.info)



 #def cutIntoLSTMSTepSizes(self,inputList):
 #  listOfList=[]
 #  for c in range(int(self.number_of_time_slices/self.number_of_lstm_time_steps)) :
 #     chunk=inputList[c:c*self.number_of_lstm_time_steps]
 #     listOfList.append(chunk)
 #  return listOfList

 def load_weights(self):
     if os.path.exists(self.model_save_dir+"/"+self.model_save_file):
         self.model.load_weights(self.model_save_dir+"/"+self.model_save_file)

 def save_weights(self):
     if not os.path.exists(self.model_save_dir):
         os.makedirs(self.model_save_dir)
     self.model.save_weights(self.model_save_dir+"/"+self.model_save_file)


 def prepareData(self,data,augment):
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  #x_data=self.mfcc(x_data)
  #x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],x_data.shape[2],1)
  x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],1)
    
  y_data=data[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded=self.uscData.one_hot_encode_array(y_data)

  #self.uscLogger.logger.info("x_data_1.shape="+str(x_data_1.shape))
  #self.uscLogger.logger.info("x_data_2.shape="+str(x_data_2.shape))
  #self.uscLogger.logger.info("y_data_one_hot_encoded_1.shape="+str(y_data_one_hot_encoded_1.shape))
  #self.uscLogger.logger.info("y_data_one_hot_encoded_2.shape="+str(y_data_one_hot_encoded_2.shape))
  #self.uscLogger.logger.info("similarity.shape="+str(similarity.shape))

  return x_data,y_data_one_hot_encoded


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 

  #self.model.fit([x_data_1,x_data_2,y_data_1,y_data_2,similarity], [y_data_1,y_data_2,x_data_1,x_data_2,similarity], epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  #self.model.fit([x_data_1,x_data_2,y_data_1,y_data_2,similarity], None, epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  
  
  #plt.plot(x_data[0])
  #plt.show()
  #self.uscData.play(x_data[0])
  
  #plt.plot(x_data_2[0])
  #plt.show()
  #self.uscData.play(20*x_data_2[0])
  
  
  
    
  #self.uscLogger.logger.info("model.train_on_batch started")
  self.model.train_on_batch([x_data],[y_data] )
  #self.uscLogger.logger.info("model.train_on_batch ended")

  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart

  #sample_weight
  evaluation = self.model.evaluate([x_data], [y_data],  batch_size = self.uscData.mini_batch_size,verbose=0)
  #prediction = self.model.predict([x_data_1,x_data_2,categorical_weight])

  #self.uscLogger.logger.info(self.model.metrics_names)
  #self.uscLogger.logger.info(evaluation)
  

  trainingLoss = evaluation[0]
  trainingAccuracy = evaluation[1]

  #self.uscLogger.logger.info("------------------------------------------------------------")
  #self.uscLogger.logger.info(np.argmax(prediction[0], axis=1))
  #self.uscLogger.logger.info(np.argmax(prediction[1], axis=1))
  #self.uscLogger.logger.info(np.argmax(prediction[2], axis=1))
  #self.uscLogger.logger.info(np.argmax(y_data_1, axis=1))
  #self.uscLogger.logger.info(np.argmax(y_data_2, axis=1))
  #self.uscLogger.logger.info(similarity)
  #self.uscLogger.logger.info(trainingAccuracy_classifier_1)
  #self.uscLogger.logger.info(trainingAccuracy_classifier_2)
  #self.uscLogger.logger.info(trainingAccuracy_discriminator)
   


  
  self.trainCount+=1
  if self.trainCount % 100 == 0 :
     self.save_weights()

  #return trainingTime,trainingLoss ,  categorical_weight[0][0][0]*trainingLoss_classifier_1 ,  categorical_weight[0][0][0]*trainingLoss_classifier_2 ,  0 ,  0 ,  trainingLoss_discriminator ,  categorical_weight[0][0][0]*trainingAccuracy_classifier_1 ,  categorical_weight[0][0][0]*trainingAccuracy_classifier_2 ,  0 ,  0 ,  trainingAccuracy_discriminator ,prepareDataTime
  return trainingTime,trainingLoss ,  0,  0 ,  0 ,  0 ,  0 ,  trainingAccuracy ,  0 ,  0 ,  0 ,  0 ,prepareDataTime
 
 def setPredictedLabel(self,data):
     x_data,y_data=self.prepareData(data,False) 
     y_pred=self.model.predict([x_data])
     # we know that similarity is real label even for youtube data, so we will use it to augment accuracy 
     predicted_value=y_pred[0]
     data[:,4*self.uscData.sound_record_sampling_rate]= np.argmax(predicted_value, axis=1)
     return data

 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment) 
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  
  
  evaluation = self.model.evaluate([x_data], [y_data],batch_size = self.uscData.mini_batch_size,verbose=0)
  y_pred=self.model.predict([x_data])
  y_pred= np.argmax(y_pred, axis=1)
  y_raw_data= np.argmax(y_data, axis=1)
  confusionMatrix=tf.math.confusion_matrix(labels=y_raw_data, predictions=y_pred,num_classes=self.uscData.number_of_classes).numpy()
 
  
  testLoss = evaluation[0]
  testAccuracy = evaluation[1]
  
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  
  return testTime,testLoss ,  0 ,  0 ,  0 ,  0 ,  0 ,  testAccuracy ,  0 ,  0 ,  0 ,  0 ,prepareDataTime,confusionMatrix
  

 def buildModel(self):
   #layer_input_1 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input_1")
   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input")
   self.uscLogger.logger.info("layer_input.shape="+str(layer_input.shape))


   # Convolution1D(filters, kernel_size,...)


   out=keras.layers.Convolution1D(16, 64,strides=16,activation='relu', padding='same')(layer_input)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(32, 32,strides=8,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.BatchNormalization()(out)
   
   
   attention=keras.layers.Convolution1D(32, 16,activation='relu', padding='same')(out)
   attention=keras.layers.Flatten()(attention)
   attention=keras.layers.Reshape((attention.shape[1],1))(attention)
   attention=keras.layers.Convolution1D(32, 64,activation='relu', padding='same')(attention)
   attention=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(attention)
   attention=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(attention)
   attention=keras.layers.Convolution1D(32, 16,strides=2,activation='relu', padding='same')(attention)
  
   
   out=keras.layers.concatenate([attention,out])
   
   out=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(32, 16,strides=4,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.Flatten()(out)
   
   first_denses=[]
   for i in range(32):
      first_denses.append(keras.layers.Dense(units = 2,activation='sigmoid')(out))
   
   second_denses=[]
   for i in range(32):
      second_denses.append(keras.layers.add([first_denses[i],first_denses[-i]]))
      
   third_denses=[]
   for i in range(32):
      third_denses.append(keras.layers.add([second_denses[i],second_denses[-i]]))
      
   fourth_denses=[]
   for i in range(32):
      fourth_denses.append(keras.layers.add([third_denses[i],third_denses[-i]]))
      
   out=keras.layers.add(fourth_denses)

   out=keras.layers.BatchNormalization()(out)
   
   out=keras.layers.Dense(units=64,activation='sigmoid')(out)
     
   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)


   
   self.model = keras.models.Model(
                          inputs=[layer_input], 
                          outputs=[out]
                          )
    
   ## categorical cross entropy = sum ( p_i * log(q_i))  , tum p_i ler  uscData.one_hot_encode_array icinde eger class number 10'u geciyorsa 0 olarak birakiliyor (class number > 10  =>  youtube data)
   ## dolayisiyla youtube data icin keras.losses.categorical_crossentropy otomatik olarak 0 gelecektir.
   
   self.model.compile(
       #optimizer=keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999),
       optimizer=keras.optimizers.Adam(lr=0.0001),
       #optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
       #loss=['categorical_crossentropy','categorical_crossentropy','mse'],
       loss=['categorical_crossentropy'],
       metrics=[['accuracy']]
   )


 def mfcc(self,data):
      stfts = tf.signal.stft(data, frame_length=4096, frame_step=1024,fft_length=4096)
      spectrograms = tf.abs(stfts)
      num_spectrogram_bins = stfts.shape[-1]
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 12000.0, 80
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                           num_mel_bins, num_spectrogram_bins, self.uscData.sound_record_sampling_rate, lower_edge_hertz,upper_edge_hertz)
      mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
      mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
      # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
      log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
      # Compute MFCCs from log_mel_spectrograms and take the first 13.
      mfccs = np.array(tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :52])
      return mfccs
 


   
   

