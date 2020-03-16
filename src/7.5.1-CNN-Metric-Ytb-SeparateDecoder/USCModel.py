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
   self.training_iterations   = 10000
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
  x_data_1=data_1[:,:4*self.uscData.sound_record_sampling_rate]
  x_data_2=data_2[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data_1=self.uscData.augment_random(x_data_1)
    x_data_2=self.uscData.augment_random(x_data_2)
  x_data_1=self.uscData.normalize(x_data_1)
  x_data_2=self.uscData.normalize(x_data_2)

  y_data_1=data_1[:,4*self.uscData.sound_record_sampling_rate]
  y_data_2=data_2[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded_1=self.uscData.one_hot_encode_array(y_data_1)
  y_data_one_hot_encoded_2=self.uscData.one_hot_encode_array(y_data_2)
  similarity=self.uscData.similarity_array(y_data_1,y_data_2)


  x_data_1=x_data_1.reshape(x_data_1.shape[0],x_data_1.shape[1],1)
  x_data_2=x_data_2.reshape(x_data_2.shape[0],x_data_2.shape[1],1)
  #y_data_one_hot_encoded_1=y_data_one_hot_encoded_1.reshape(y_data_one_hot_encoded_1.shape[0],y_data_one_hot_encoded_1.shape[1],1)
  #y_data_one_hot_encoded_2=y_data_one_hot_encoded_2.reshape(y_data_one_hot_encoded_2.shape[0],y_data_one_hot_encoded_2.shape[1],1)
  similarity=similarity.reshape(similarity.shape[0],1)


  #self.uscLogger.logger.info("x_data_1.shape="+str(x_data_1.shape))
  #self.uscLogger.logger.info("x_data_2.shape="+str(x_data_2.shape))
  #self.uscLogger.logger.info("y_data_one_hot_encoded_1.shape="+str(y_data_one_hot_encoded_1.shape))
  #self.uscLogger.logger.info("y_data_one_hot_encoded_2.shape="+str(y_data_one_hot_encoded_2.shape))
  #self.uscLogger.logger.info("similarity.shape="+str(similarity.shape))

  return x_data_1,x_data_2,y_data_one_hot_encoded_1,y_data_one_hot_encoded_2,similarity


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data_1,x_data_2,y_data_1,y_data_2,similarity=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 

  #self.model.fit([x_data_1,x_data_2,y_data_1,y_data_2,similarity], [y_data_1,y_data_2,x_data_1,x_data_2,similarity], epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  #self.model.fit([x_data_1,x_data_2,y_data_1,y_data_2,similarity], None, epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  
  #self.uscLogger.logger.info("model.train_on_batch started")
  self.model.train_on_batch([x_data_1,x_data_2],[y_data_1,y_data_2,x_data_1,x_data_2,similarity] )
  #self.uscLogger.logger.info("model.train_on_batch ended")

  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart

  evaluation = self.model.evaluate([x_data_1,x_data_2], [y_data_1,y_data_2,x_data_1,x_data_2,similarity], batch_size = self.uscData.mini_batch_size,verbose=0)

  #self.uscLogger.logger.info(self.model.metrics_names)
  #self.uscLogger.logger.info(evaluation)
  

  trainingLoss = evaluation[0]
  trainingLoss_classifier_1 = evaluation[1]
  trainingLoss_classifier_2 = evaluation[2]
  trainingLoss_autoencoder_1 = evaluation[3]
  trainingLoss_autoencoder_2 = evaluation[4]
  trainingLoss_discriminator = evaluation[5]

  trainingAccuracy_classifier_1 = evaluation[6]
  trainingAccuracy_classifier_2 = evaluation[7]
  trainingAccuracy_autoencoder_1 = evaluation[8]
  trainingAccuracy_autoencoder_2 = evaluation[9]
  trainingAccuracy_discriminator = evaluation[10]

  
  self.trainCount+=1
  if self.trainCount % 100 == 0 :
     self.save_weights()

  return trainingTime,trainingLoss ,  trainingLoss_classifier_1 ,  trainingLoss_classifier_2 ,  trainingLoss_autoencoder_1 ,  trainingLoss_autoencoder_2 ,  trainingLoss_discriminator ,  trainingAccuracy_classifier_1 ,  trainingAccuracy_classifier_2 ,  trainingAccuracy_autoencoder_1 ,  trainingAccuracy_autoencoder_2 ,  trainingAccuracy_discriminator ,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  prepareDataTimeStart = int(round(time.time())) 
  x_data_1,x_data_2,y_data_1,y_data_2,similarity=self.prepareData(data,augment) 
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  
  
  evaluation = self.model.evaluate([x_data_1,x_data_2], [y_data_1,y_data_2,x_data_1,x_data_2,similarity],batch_size = self.uscData.mini_batch_size,verbose=0)
  testLoss = evaluation[0]
  testLoss_classifier_1 = evaluation[1]
  testLoss_classifier_2 = evaluation[2]
  testLoss_autoencoder_1 = evaluation[3]
  testLoss_autoencoder_2 = evaluation[4]
  testLoss_discriminator = evaluation[5]

  testAccuracy_classifier_1 = evaluation[6]
  testAccuracy_classifier_2 = evaluation[7]
  testAccuracy_autoencoder_1 = evaluation[8]
  testAccuracy_autoencoder_2 = evaluation[9]
  testAccuracy_discriminator = evaluation[10]
  
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  
  return testTime,testLoss ,  testLoss_classifier_1 ,  testLoss_classifier_2 ,  testLoss_autoencoder_1 ,  testLoss_autoencoder_2 ,  testLoss_discriminator ,  testAccuracy_classifier_1 ,  testAccuracy_classifier_2 ,  testAccuracy_autoencoder_1 ,  testAccuracy_autoencoder_2 ,  testAccuracy_discriminator ,prepareDataTime
  

 def buildModel(self):

   layer_input_1 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input_1")
   self.uscLogger.logger.info("layer_input_1.shape="+str(layer_input_1.shape))

   layer_input_2 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input_2")
   self.uscLogger.logger.info("layer_input_2.shape="+str(layer_input_2.shape))

   layer_input=keras.layers.concatenate([layer_input_1,layer_input_2])
   self.uscLogger.logger.info("layer_input.shape="+str(layer_input.shape))
   
   # Convolution1D(filters, kernel_size,...)
   out=keras.layers.Convolution1D(16, 64,strides=25,activation='relu', padding='same')(layer_input)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(64,32,strides=7,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(64, 16,strides=7,activation='relu', padding='same')(out)
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(out)
   out=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(out)
   out=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(out)
   out=keras.layers.BatchNormalization()(out)
   
   common_cnn_out=out
   self.uscLogger.logger.info("encoder.shape="+str(out.shape))

   classifier_out_1=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(common_cnn_out)
   classifier_out_1=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(classifier_out_1)
   classifier_out_1=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(classifier_out_1)
   classifier_out_1=keras.layers.Flatten()(classifier_out_1)
   classifier_out_1=keras.layers.Dense(units = 512,activation='sigmoid')(classifier_out_1)
   classifier_out_1=keras.layers.BatchNormalization()(classifier_out_1)
   classifier_out_1=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out_1)

   classifier_out_2=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(common_cnn_out)
   classifier_out_2=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(classifier_out_2)
   classifier_out_2=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(classifier_out_2)
   classifier_out_2=keras.layers.Flatten()(classifier_out_2)
   classifier_out_2=keras.layers.Dense(units = 512,activation='sigmoid')(classifier_out_2)
   classifier_out_2=keras.layers.BatchNormalization()(classifier_out_2)
   classifier_out_2=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out_2)
   

   autoencoder_out_1=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_cnn_out)
   autoencoder_out_1=keras.layers.UpSampling1D(2)(autoencoder_out_1)
   autoencoder_out_1=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(autoencoder_out_1)
   autoencoder_out_1=keras.layers.UpSampling1D(2)(autoencoder_out_1)
   autoencoder_out_1=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(autoencoder_out_1)
   autoencoder_out_1=keras.layers.UpSampling1D(2)(autoencoder_out_1)
   autoencoder_out_1=keras.layers.BatchNormalization()(autoencoder_out_1)
   autoencoder_out_1=keras.layers.Convolution1D(64,16, activation='relu', padding='same')(autoencoder_out_1)
   autoencoder_out_1=keras.layers.UpSampling1D(7)(autoencoder_out_1)
   autoencoder_out_1=keras.layers.Convolution1D(64,32, activation='relu', padding='same')(autoencoder_out_1)
   autoencoder_out_1=keras.layers.UpSampling1D(7)(autoencoder_out_1)
   autoencoder_out_1=keras.layers.Convolution1D(16,64, activation='relu', padding='same')(autoencoder_out_1)
   autoencoder_out_1=keras.layers.UpSampling1D(25)(autoencoder_out_1)
   autoencoder_out_1=keras.layers.Convolution1D(1,64,activation='sigmoid', padding='same')(autoencoder_out_1)


   autoencoder_out_2=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_cnn_out)
   autoencoder_out_2=keras.layers.UpSampling1D(2)(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(autoencoder_out_2)
   autoencoder_out_2=keras.layers.UpSampling1D(2)(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(autoencoder_out_2)
   autoencoder_out_2=keras.layers.UpSampling1D(2)(autoencoder_out_2)
   autoencoder_out_2=keras.layers.BatchNormalization()(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Convolution1D(64,16, activation='relu', padding='same')(autoencoder_out_2)
   autoencoder_out_2=keras.layers.UpSampling1D(7)(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Convolution1D(64,32, activation='relu', padding='same')(autoencoder_out_2)
   autoencoder_out_2=keras.layers.UpSampling1D(7)(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Convolution1D(16,64, activation='relu', padding='same')(autoencoder_out_2)
   autoencoder_out_2=keras.layers.UpSampling1D(25)(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Convolution1D(1,64,activation='sigmoid', padding='same')(autoencoder_out_2)









   discriminator_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_cnn_out)
   discriminator_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(discriminator_out)
   discriminator_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(discriminator_out)
   discriminator_out=keras.layers.Flatten()(discriminator_out)
   discriminator_out=keras.layers.Dense(units = 512,activation='sigmoid')(discriminator_out)
   discriminator_out=keras.layers.BatchNormalization()(discriminator_out)
   discriminator_out=keras.layers.Dense(units = 1,activation='softmax')(discriminator_out)
   

   self.uscLogger.logger.info("classifier_out_1.shape="+str(classifier_out_1.shape))
   self.uscLogger.logger.info("classifier_out_2.shape="+str(classifier_out_2.shape))
   self.uscLogger.logger.info("layer_input_1.shape="+str(layer_input_1.shape))
   self.uscLogger.logger.info("layer_input_2.shape="+str(layer_input_2.shape))
   self.uscLogger.logger.info("autoencoder_out_1.shape="+str(autoencoder_out_1.shape))
   self.uscLogger.logger.info("autoencoder_out_2.shape="+str(autoencoder_out_2.shape))
   self.uscLogger.logger.info("discriminator_out.shape="+str(discriminator_out.shape))
   
   self.model = keras.models.Model(
                          inputs=[layer_input_1,layer_input_2], 
                          outputs=[classifier_out_1,classifier_out_2,autoencoder_out_1,autoencoder_out_2,discriminator_out]
                          )
    
   ## categorical cross entropy = sum ( p_i * log(q_i))  , tum p_i ler  uscData.one_hot_encode_array icinde eger class number 10'u geciyorsa 0 olarak birakiliyor (class number > 10  =>  youtube data)
   ## dolayisiyla youtube data icin keras.losses.categorical_crossentropy otomatik olarak 0 gelecektir.
   
   self.model.compile(
       optimizer=keras.optimizers.Adam(lr=0.00001),
       loss=['categorical_crossentropy','categorical_crossentropy','mse','mse','mse'],
       loss_weights=[4/20,   4/20,   4/20,   4/20,   4/20],
       metrics=[['accuracy'],['accuracy'],['accuracy'],['accuracy'],['accuracy']]
   )






   
   

