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
  data_1=data
  data_2=np.random.permutation(data)
  if np.random.rand() > 0.5 :
     data_2[::2]=data_1[::2] ## equalize elements with even indices ,  this gurarantess that 50% is the same
  else :
     data_2[1::2]=data_1[1::2] ## equalize elements with odd indices ,   this gurarantess that  50% is the same


  x_data_1=data_1[:,:4*self.uscData.sound_record_sampling_rate]
  x_data_2=data_2[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data_1=self.uscData.augment_random(x_data_1)
    x_data_2=self.uscData.augment_random(x_data_2)
  x_data_1=self.uscData.normalize(x_data_1)
  x_data_2=self.uscData.normalize(x_data_2)

  x_data_1=self.mfcc(x_data_1)
  x_data_2=self.mfcc(x_data_2)
  x_data_1=x_data_1.reshape(x_data_1.shape[0],x_data_1.shape[1],x_data_1.shape[2],1)
  x_data_2=x_data_2.reshape(x_data_2.shape[0],x_data_2.shape[1],x_data_1.shape[2],1)
  
  
  y_data_1=data_1[:,4*self.uscData.sound_record_sampling_rate]
  y_data_2=data_2[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded_1=self.uscData.one_hot_encode_array(y_data_1)
  y_data_one_hot_encoded_2=self.uscData.one_hot_encode_array(y_data_2)
  similarity=self.uscData.similarity_array(y_data_1,y_data_2)



  #y_data_one_hot_encoded_1=y_data_one_hot_encoded_1.reshape(y_data_one_hot_encoded_1.shape[0],y_data_one_hot_encoded_1.shape[1],1)
  #y_data_one_hot_encoded_2=y_data_one_hot_encoded_2.reshape(y_data_one_hot_encoded_2.shape[0],y_data_one_hot_encoded_2.shape[1],1)
  similarity=similarity.reshape(similarity.shape[0],1)


  #self.uscLogger.logger.info("x_data_1.shape="+str(x_data_1.shape))
  #self.uscLogger.logger.info("x_data_2.shape="+str(x_data_2.shape))
  #self.uscLogger.logger.info("y_data_one_hot_encoded_1.shape="+str(y_data_one_hot_encoded_1.shape))
  #self.uscLogger.logger.info("y_data_one_hot_encoded_2.shape="+str(y_data_one_hot_encoded_2.shape))
  #self.uscLogger.logger.info("similarity.shape="+str(similarity.shape))

  return x_data_1,x_data_2,y_data_one_hot_encoded_1,y_data_one_hot_encoded_2,similarity


 def train(self,data,categorical_weight):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data_1,x_data_2,y_data_1,y_data_2,similarity=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 

  #self.model.fit([x_data_1,x_data_2,y_data_1,y_data_2,similarity], [y_data_1,y_data_2,x_data_1,x_data_2,similarity], epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  #self.model.fit([x_data_1,x_data_2,y_data_1,y_data_2,similarity], None, epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  
  
  #plt.plot(x_data_1[0])
  #plt.show()
  #self.uscData.play(x_data_1[0])
  
  #plt.plot(x_data_2[0])
  #plt.show()
  #self.uscData.play(20*x_data_2[0])
  
  
  
    
  #self.uscLogger.logger.info("model.train_on_batch started")
  self.model.train_on_batch([x_data_1,x_data_2,categorical_weight],[y_data_1,y_data_2,similarity] )
  #self.uscLogger.logger.info("model.train_on_batch ended")

  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart

  #sample_weight
  evaluation = self.model.evaluate([x_data_1,x_data_2,categorical_weight], [y_data_1,y_data_2,similarity],  batch_size = self.uscData.mini_batch_size,verbose=0)
  #prediction = self.model.predict([x_data_1,x_data_2,categorical_weight])

  #self.uscLogger.logger.info(self.model.metrics_names)
  #self.uscLogger.logger.info(evaluation)
  

  trainingLoss = evaluation[0]
  trainingLoss_classifier_1 = evaluation[1]
  trainingLoss_classifier_2 = evaluation[2]
  trainingLoss_discriminator = evaluation[3]
  trainingAccuracy_classifier_1 = evaluation[4]
  trainingAccuracy_classifier_2 = evaluation[5]
  trainingAccuracy_discriminator = evaluation[6]

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
  return trainingTime,trainingLoss ,  trainingLoss_classifier_1 ,  trainingLoss_classifier_2 ,  0 ,  0 ,  trainingLoss_discriminator ,  trainingAccuracy_classifier_1 ,  trainingAccuracy_classifier_2 ,  0 ,  0 ,  trainingAccuracy_discriminator ,prepareDataTime
 
 def setPredictedLabel(self,data,categorical_weight):
     x_data_1,x_data_2,y_data_1,y_data_2,similarity=self.prepareData(data,False) 
     y_pred=self.model.predict([x_data_1,x_data_2,categorical_weight])
     # we know that similarity is real label even for youtube data, so we will use it to augment accuracy 
     if data[0][0]> 0 : # cheap randomness 50%, (may be the other prediction is true :) )
         predicted_value=y_pred[0]+y_pred[1]*similarity
         ## how to use similarity ?:
         ## if similar :
         ##   if y_pred_0 != y_pred_1 :
         ##        wrong prediction , then use arg_max(y_pred_0+y_pred_1)
         ##   else :
         ##        at least the prediction obeys the similarity, so again use argmax(y_pred_0+y_pred_1)
         ## if not similar:
         ##   use y_pred_0
     else :
         predicted_value=y_pred[1]+y_pred[0]*similarity

     data[:,4*self.uscData.sound_record_sampling_rate]= np.argmax(predicted_value, axis=1)
     return data

 def test(self,data,categorical_weight):
  testTimeStart = int(round(time.time())) 
  augment=False
  prepareDataTimeStart = int(round(time.time())) 
  x_data_1,x_data_2,y_data_1,y_data_2,similarity=self.prepareData(data,augment) 
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  
  
  evaluation = self.model.evaluate([x_data_1,x_data_2,categorical_weight], [y_data_1,y_data_2,similarity],batch_size = self.uscData.mini_batch_size,verbose=0)
  y_pred=self.model.predict([x_data_1,x_data_2,categorical_weight])
  y_pred= np.argmax(y_pred[0], axis=1)
  y_raw_data_1= np.argmax(y_data_1, axis=1)
  confusionMatrix=tf.math.confusion_matrix(labels=y_raw_data_1, predictions=y_pred,num_classes=self.uscData.number_of_classes).numpy()
 
  
  testLoss = evaluation[0]
  testLoss_classifier_1 = evaluation[1]
  testLoss_classifier_2 = evaluation[2]
  testLoss_discriminator = evaluation[3]
  testAccuracy_classifier_1 = evaluation[4]
  testAccuracy_classifier_2 = evaluation[5]
  testAccuracy_discriminator = evaluation[6]
  
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  
  return testTime,testLoss ,  testLoss_classifier_1 ,  testLoss_classifier_2 ,  0 ,  0 ,  testLoss_discriminator ,  testAccuracy_classifier_1 ,  testAccuracy_classifier_2 ,  0 ,  0 ,  testAccuracy_discriminator ,prepareDataTime,confusionMatrix
  

 def buildModel(self):
   #layer_input_1 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input_1")
   layer_input_1 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,686,13,1),name="layer_input_1")
   self.uscLogger.logger.info("layer_input_1.shape="+str(layer_input_1.shape))

   #layer_input_2 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input_2")
   layer_input_2 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,686,13,1),name="layer_input_2")
   self.uscLogger.logger.info("layer_input_2.shape="+str(layer_input_2.shape))

   layer_categorical_weight = keras.layers.Input(shape=(1,1),name="layer_categorical_weight")
   self.uscLogger.logger.info("layer_categorical_weight.shape="+str(layer_categorical_weight.shape))

   layer_input=keras.layers.concatenate([layer_input_1,layer_input_2],axis=1)
   self.uscLogger.logger.info("layer_input.shape="+str(layer_input.shape))
   
   # Convolution1D(filters, kernel_size,...)


   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(layer_input)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,2), padding='same')(out)
     
   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,2), padding='same')(out)
   
   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,1), padding='same')(out)
   
   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,1), padding='same')(out)
   
   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,1), padding='same')(out)
   
   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,1), padding='same')(out)
   
   out=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.MaxPooling2D((3,3),strides=(2,1), padding='same')(out)
   

   out=keras.layers.BatchNormalization()(out)

   common_cnn_out=out

   self.uscLogger.logger.info("common_cnn_out.shape="+str(common_cnn_out.shape))

   classifier_out_1=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(common_cnn_out)
   classifier_out_1=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(classifier_out_1)
   classifier_cnn_out_1=classifier_out_1
   classifier_out_1_flat=keras.layers.Flatten()(classifier_out_1)
   classifier_out_1=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_1_flat)
   classifier_out_1=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_1)
   classifier_out_1=keras.layers.BatchNormalization()(classifier_out_1)
   classifier_out_1=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out_1)

   classifier_out_2=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(common_cnn_out)
   classifier_out_2=keras.layers.Convolution2D(64, (3, 3),activation='relu', padding='same')(classifier_out_2)
   classifier_cnn_out_2=classifier_out_2
   classifier_out_2_flat=keras.layers.Flatten()(classifier_out_2)
   classifier_out_2=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_2_flat)
   classifier_out_2=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_2)
   classifier_out_2=keras.layers.BatchNormalization()(classifier_out_2)
   classifier_out_2=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out_2)


   classifier_out=keras.layers.concatenate([classifier_cnn_out_1,classifier_cnn_out_2])
   #classifier_out=common_cnn_out

   discriminator_out=keras.layers.Convolution2D(64, (3, 3),activation='relu')(classifier_out)
   discriminator_out=keras.layers.Flatten()(discriminator_out)
   discriminator_out=keras.layers.Dense(units = 64,activation='sigmoid')(discriminator_out)
   discriminator_out=keras.layers.BatchNormalization()(discriminator_out)
   discriminator_out=keras.layers.Dense(units = 1,activation='sigmoid')(discriminator_out)
   

   self.uscLogger.logger.info("classifier_out_1.shape="+str(classifier_out_1.shape))
   self.uscLogger.logger.info("classifier_out_2.shape="+str(classifier_out_2.shape))
   self.uscLogger.logger.info("layer_input_1.shape="+str(layer_input_1.shape))
   self.uscLogger.logger.info("layer_input_2.shape="+str(layer_input_2.shape))
   self.uscLogger.logger.info("discriminator_out.shape="+str(discriminator_out.shape))
   
   self.model = keras.models.Model(
                          inputs=[layer_input_1,layer_input_2,layer_categorical_weight], 
                          outputs=[classifier_out_1,classifier_out_2,discriminator_out]
                          )
    
   ## categorical cross entropy = sum ( p_i * log(q_i))  , tum p_i ler  uscData.one_hot_encode_array icinde eger class number 10'u geciyorsa 0 olarak birakiliyor (class number > 10  =>  youtube data)
   ## dolayisiyla youtube data icin keras.losses.categorical_crossentropy otomatik olarak 0 gelecektir.
   
   self.model.compile(
       #optimizer=keras.optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999),
       optimizer=keras.optimizers.Adam(lr=0.0001),
       #optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
       #loss=['categorical_crossentropy','categorical_crossentropy','mse'],
       loss=['categorical_crossentropy','categorical_crossentropy','binary_crossentropy'],
       loss_weights=[layer_categorical_weight*8/20,   layer_categorical_weight*8/20,   4/20],
       metrics=[['accuracy'],['accuracy'],['accuracy']]
   )


 def mfcc(self,data):
      stfts = tf.signal.stft(data, frame_length=1024, frame_step=256,fft_length=1024)
      spectrograms = tf.abs(stfts)
      num_spectrogram_bins = stfts.shape[-1]
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000.0, 80
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                           num_mel_bins, num_spectrogram_bins, self.uscData.sound_record_sampling_rate, lower_edge_hertz,upper_edge_hertz)
      mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
      mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
      # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
      log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
      # Compute MFCCs from log_mel_spectrograms and take the first 13.
      mfccs = np.array(tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13])
      return mfccs
 


   
   

