#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class NeuralNetworkModel :
 def __init__(self, session, logger, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE , mini_batch_size=MINI_BATCH_SIZE, time_slice_length=TIME_SLICE_LENGTH,time_slice_overlap_length=TIME_SLICE_OVERLAP_LENGTH,number_of_lstm_time_steps=MIN_NUMBER_OF_LSTM_TIME_STEPS_TO_RECOGNIZE_TRACK_CLASS,lstm_size=LSTM_SIZE):
   self.session               = session
   self.logger                = logger
   self.input_size            = input_size
   self.output_size           = output_size
   self.mini_batch_size       = mini_batch_size
   self.time_slice_length     = time_slice_length
   self.time_slice_overlap_length= time_slice_overlap_length
   self.number_of_lstm_time_steps=number_of_lstm_time_steps
   self.lstm_size=lstm_size

   self.real_y_values=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values")
   self.x_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   self.x_input_list_list=[]


   a=self.x_input.shape[1]-self.time_slice_length
   b=self.time_slice_length-self.time_slice_overlap_length
   self.number_of_time_slices=int(a)/int(b)+1
   with tf.name_scope('input_overlapped_reshape'):
    logger.info("self.x_input.shape="+str(self.x_input.shape))
    stride=time_slice_length-time_slice_overlap_length
    reshaped = tf.reshape(self.x_input, [self.mini_batch_size,1,-1,1])
    ones = tf.ones(time_slice_length, dtype=tf.float32)
    ident = tf.diag(ones)
    filter_dim = [1, time_slice_length, time_slice_length, 1]
    filter_matrix = tf.reshape(ident, filter_dim)
    stride_window = [1, 1, stride, 1]
    filtered_conv = []
    for f in tf.unstack(filter_matrix, axis=1):
      reshaped_filter = tf.reshape(f, [1, time_slice_length, 1, 1])
      c = tf.nn.conv2d(reshaped, reshaped_filter, stride_window, padding='VALID')
      filtered_conv.append(c)
    t = tf.stack(filtered_conv, axis=3)
    self.x_input_reshaped = tf.squeeze(t)
    logger.info("self.number_of_time_slices="+str(self.number_of_time_slices))
    self.x_input_list = tf.unstack(self.x_input_reshaped, self.number_of_time_slices, 1)
        ###  HANN WINDOW + FFT + NORMALIZE

   with tf.name_scope('input_fft'):
    windowed_frame = self.x_input_list * tf.contrib.signal.hann_window(time_slice_length, periodic=True)
    windowed_frame_FFT=tf.abs(tf.signal.fft(windowed_frame))
    windowed_frame_FFT_Normalized=tf.keras.utils.normalize(windowed_frame_FFT,axis=1)
    self.x_input_list=windowed_frame_FFT_Normalized
    ####
    logger.info("self.x_input[0].shape"+str(self.x_input_list[0].shape))

   with tf.name_scope('cut_x_input_list_into_lstm_time_steps'):
       for c in range(int(self.number_of_time_slices/self.number_of_lstm_time_steps)) :
         chunk=self.x_input_list[c:c*self.number_of_lstm_time_steps]
         self.x_input_list_list.append(chunk)

   buildModel(self.x_input_reshaped)


    

 def prepareData(self,data,augment):
  x_data=data[:,:4*SOUND_RECORD_SAMPLING_RATE]
  if augment==True :
    x_data=augment_random(x_data)
  y_data=data[:,4*SOUND_RECORD_SAMPLING_RATE]
  y_data_one_hot_encoded=one_hot_encode_array(y_data)
  return x_data,y_data_one_hot_encoded






 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 

  self.model.fit(x_data, y_data, epochs = 1, batch_size = self.mini_batch_size,verbose=2)
  
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  evaluation = self.model.evaluate(x_data, y_data)
  trainingAccuracy = evaluation['acc']
  return trainingTime,trainingAccuracy,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data,y_data=self.prepareData(data,augment) 
  evaluation = self.model.evaluate(x_data, y_data)
  testAccuracy = evaluation['acc']
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  

 def buildModel(inputTensor): 
   self.model = keras.models.Sequential()
   self.model.add(keras.layers.LSTM(self.lstm_size,dropout=0.2,recurrent_dropout=0.2,input_shape=inputTensor.shape)) 
   self.model.add(keras.layers.Dropout(0.2))
   self.model.add(keras.layers.Dense(units = 1024))
   ## self.output_size == number of classes (which is 10 in our case)
   self.model.add(keras.layers.Dense(self.output_size, activation=tf.nn.softmax))
   self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])



