#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *

##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class USCModel :
 def __init__(self, session, uscLogger,uscData): 
   self.session               = session
   self.uscLogger             = uscLogger
   self.uscData               = uscData
   self.mini_batch_size       = 100
   self.lstm_size             = 2
   #self.lstm_time_steps       = 20
   self.training_iterations   = 1000
   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True

   self.real_y_values=tf.placeholder(tf.float32, shape=(self.uscData.mini_batch_size, self.uscData.number_of_classes), name="real_y_values")
   self.x_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.uscData.track_length), name="input")
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
  x_data=self.uscData.overlapping_hanning_slice(x_data)
  x_data=self.uscData.fft(x_data)
  x_data=self.uscData.normalize(x_data)
  y_data=data[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded=self.uscData.one_hot_encode_array(y_data)
  return x_data,y_data_one_hot_encoded


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  self.model.fit(x_data, y_data, epochs = 1, batch_size = self.mini_batch_size,verbose=0)
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  evaluation = self.model.evaluate(x_data, y_data, batch_size = self.mini_batch_size)
  trainingAccuracy = evaluation[1]
  #print(self.model.metrics_names) 
  #print(evaluation) 
  return trainingTime,trainingAccuracy,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data,y_data=self.prepareData(data,augment) 
  evaluation = self.model.evaluate(x_data, y_data,batch_size = self.mini_batch_size)
  testAccuracy = evaluation[1]
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  
 def buildModel(self):
   self.model = keras.models.Sequential()
   self.model.add(keras.layers.LSTM(self.lstm_size,dropout=0.2,recurrent_dropout=0.2,batch_input_shape=(self.mini_batch_size,self.uscData.number_of_time_slices,self.uscData.time_slice_length)))
   self.model.add(keras.layers.Dropout(0.2))
   self.model.add(keras.layers.Dense(units = 1024))
   ## self.output_size == number of classes (which is 10 in our case)
   self.model.add(keras.layers.Dense(self.uscData.number_of_classes, activation=tf.nn.softmax))
   self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])




