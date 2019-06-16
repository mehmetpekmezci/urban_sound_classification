#!/usr/bin/env python3
from header import *
from data import *


##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class NeuralNetworkModel :

    

    
 def __init__(self, session, logger,autoEncoder):
   self.session               = session
   self.logger                = logger
   self.autoEncoder           = autoEncoder
  
   ##
   ## DEFINE PLACE HOLDER FOR REAL OUTPUT VALUES FOR TRAINING
   ##
   self.real_y_values=tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values")
   self.x_input = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.input_size), name="input")
   # x_input.shape[1]=time_slice_length+ number_of_time_slices * (time_slice_length-time_slice_overlap_length) 

   
   ##
   ## LSTM LAYERS
   ##
   with tf.name_scope("lstm"):
     self.model = Sequential()
     self.model.add(LSTM(1024,dropout=0.5,recurrent_dropout=0.5,input_shape=(batchSize,LSTM_TIME_STEPS,dimensionsOfData)))
     self.model.add(Dropout(0.5))
     self.model.add(Dense(units = 2048))
     self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy') 
    

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
  #self.logger.info('self.session.run(self.x_input_list[0][0][self.time_slice_length-self.time_slice_overlap_length],feed_dict={self.x_input: x_data})=')
  #self.logger.info(self.session.run(self.x_input_list[0][0][self.time_slice_length-self.time_slice_overlap_length],feed_dict={self.x_input: x_data}))
  #self.logger.info('self.session.run(self.x_input_list[1][0][0],feed_dict={self.x_input: x_data})=')
  #self.logger.info(self.session.run(self.x_input_list[1][0][0],feed_dict={self.x_input: x_data}))
  self.optimizer.run(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob:self.keep_prob_constant})
  
  
  
  
  
  
    trainingInputFrames=[] # list of list (LSTM_TIME_STEPS) of 20*3=60 data points
  trainingOutputFrames=[] # list of 20*3=60 data points
  self.model.fit(trainingInputFrames, trainingOutputFrames, epochs = 10, batch_size =MINI_BATCH_SIZE,verbose=2)
  predictions=model.predict(testInputFrames)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingAccuracy = self.categorical_accuracy(y_data,predictions)
  return trainingTime,trainingAccuracy,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data,y_data=self.prepareData(data,augment) 
  testAccuracy = self.accuracy.eval(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob: 1.0})
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  


 def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                          K.floatx()) 


 def convertToOverlappingSequentialData(x_input):
   self.logger.info("self.x_input.shape="+str(self.x_input.shape))
   stride=TIME_SLICE_LENGTH-TIME_SLICE_OVERLAP_LENGTH
   reshaped = tf.reshape(self.x_input, [MINI_BATCH_SIZE,1,-1,1])
   ones = tf.ones(TIME_SLICE_LENGTH, dtype=tf.float32)
   ident = tf.diag(ones)
   filter_dim = [1, TIME_SLICE_LENGTH, TIME_SLICE_LENGTH, 1]
   filter_matrix = tf.reshape(ident, filter_dim)
   stride_window = [1, 1, stride, 1]
   filtered_conv = []
   for f in tf.unstack(filter_matrix, axis=1):
      reshaped_filter = tf.reshape(f, [1, TIME_SLICE_LENGTH, 1, 1])
      c = tf.nn.conv2d(reshaped, reshaped_filter, stride_window, padding='VALID')
      filtered_conv.append(c)
   t = tf.stack(filtered_conv, axis=3)
   self.x_input_reshaped = tf.squeeze(t)
   self.logger.info("self.number_of_time_slices="+str(NUMBER_OF_TIME_SLICES))
   x_input_list = tf.unstack(self.x_input_reshaped,NUMBER_OF_TIME_SLICES, 1)
   self.logger.info("self.x_input[0].shape"+str(self.x_input_list[0].shape))
   return x_input_list