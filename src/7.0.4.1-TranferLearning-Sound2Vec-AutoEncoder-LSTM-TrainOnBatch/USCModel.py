#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *

##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class USCModel :
 def __init__(self,  uscLogger,uscData,uscAutoEncoder): 
   self.uscLogger             = uscLogger
   self.uscData               = uscData
   self.uscAutoEncoder        = uscAutoEncoder
   self.lstm_size             = 1024
   ## self.uscData.time_slice_length = 440
   ## so we will have nearly 400 time steps in 4 secs record. (88200) (with %50 overlapping)
   self.lstm_time_steps       = self.uscData.number_of_time_slices
   self.training_iterations   = 10000
   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True
   self.buildModel()

 #def cutIntoLSTMSTepSizes(self,inputList):
 #  listOfList=[]
 #  for c in range(int(self.number_of_time_slices/self.number_of_lstm_time_steps)) :
 #     chunk=inputList[c:c*self.number_of_lstm_time_steps]
 #     listOfList.append(chunk)
 #  return listOfList


 def prepareData(self,data,augment):
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  x_data=self.uscData.overlapping_slice(x_data) #  (self.mini_batch_size,self.number_of_time_slices,self.time_slice_length)
  #x_data=self.uscData.fft(x_data)
  #x_data_list = self.uscData.convert_to_list_for_parallel_lstms(x_data,self.num_of_paralel_lstms,self.lstm_time_steps)
  #self.uscLogger.logger.info("x_data.shape="+str(x_data.shape))
  encodedValue,encodeTime=self.uscAutoEncoder.encode(x_data)  # (self.uscData.mini_batch_size,self.uscData.number_of_time_slices,self.latent_space_presentation_data_length)
  #self.uscLogger.logger.info("encodedValue.shape="+str(encodedValue.shape))
  y_data=data[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded=self.uscData.one_hot_encode_array(y_data)
  return encodedValue,y_data_one_hot_encoded


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  self.model.train_on_batch(x_data, y_data)
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  evaluation = self.model.evaluate(x_data, y_data, batch_size = self.uscData.mini_batch_size,verbose=0)
  trainingAccuracy = evaluation[1]
  #print(self.model.metrics_names) 
  #print(evaluation) 
  return trainingTime,trainingAccuracy,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data,y_data=self.prepareData(data,augment) 
  evaluation = self.model.evaluate(x_data, y_data,batch_size = self.uscData.mini_batch_size,verbose=0)
  testAccuracy = evaluation[1]
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  
 def buildModel(self):
   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.number_of_time_slices,self.uscData.latent_space_presentation_data_length))
   out=keras.layers.LSTM(self.lstm_size,dropout=0.2,recurrent_dropout=0.2)(layer_input)
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.Dense(units = 2048,activation='relu')(out)
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Dense(units = 2048,activation='relu')(out)
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)
   self.model = keras.models.Model(inputs=[layer_input], outputs=[out])
   selectedOptimizer=keras.optimizers.Adam(lr=0.0001)
   self.model.compile(optimizer=selectedOptimizer, loss='categorical_crossentropy',metrics=['accuracy'])


