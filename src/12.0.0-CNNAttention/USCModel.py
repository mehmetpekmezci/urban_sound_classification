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
  data=np.random.permutation(data)
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  if augment==True :
    x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  y_data=data[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded=self.uscData.one_hot_encode_array(y_data)
  x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],1)
  return x_data,y_data_one_hot_encoded


 def train(self,data,categorical_weight):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  self.model.train_on_batch([x_data],[y_data] )
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  evaluation = self.model.evaluate([x_data], [y_data], batch_size = self.uscData.mini_batch_size,verbose=0)

  #self.uscLogger.logger.info(self.model.metrics_names)
  #self.uscLogger.logger.info(evaluation)
  

  trainingLoss = evaluation[0]
  trainingAccuracy = evaluation[1]
  
  self.trainCount+=1
  if self.trainCount % 100 == 0 :
     self.save_weights()

  return trainingTime,trainingLoss , trainingAccuracy ,prepareDataTime
     
 def test(self,data,categorical_weight):
  testTimeStart = int(round(time.time())) 
  augment=False
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment) 
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  evaluation = self.model.evaluate([x_data], [y_data],batch_size = self.uscData.mini_batch_size,verbose=0)

  testLoss = evaluation[0]
  testAccuracy = evaluation[1]
  
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  
  return testTime,testLoss , testAccuracy ,prepareDataTime
  

 def buildModel(self):

   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input")
   self.uscLogger.logger.info("layer_input.shape="+str(layer_input.shape))
   # Convolution1D(filters, kernel_size,...)
   attention=keras.layers.Convolution1D(4, 128,activation='relu', padding='same')(layer_input)
   attention=keras.layers.Convolution1D(4, 64,activation='relu', padding='same')(attention)
   attention=keras.layers.Convolution1D(4, 16,activation='relu', padding='same')(attention)
   attention=keras.layers.Convolution1D(4, 64,activation='relu', padding='same')(attention)
   attention=keras.layers.Convolution1D(1, 128,activation='sigmoid', padding='same')(attention)

   self.uscLogger.logger.info("attention.shape="+str(attention.shape))

   layer_input_2=keras.layers.add([layer_input,attention])
   #layer_input_2=layer_input
   self.uscLogger.logger.info("layer_input_2.shape="+str(layer_input_2.shape))

   out=keras.layers.Convolution1D(256, 128,strides=49,activation='relu', padding='same')(layer_input_2)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(16,32,strides=10,activation='relu', padding='same')(out)
   out=keras.layers.Dropout(0.2)(out)
   out=keras.layers.Convolution1D(16, 16,strides=6,activation='relu', padding='same')(out)
   out=keras.layers.Convolution1D(16, 8,strides=2,activation='relu', padding='same')(out)
   #out_start=out
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   out=keras.layers.add([out,keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)])# residuals
   #out=keras.layers.add([out,out_start]) # first to last residual
   out=keras.layers.BatchNormalization()(out)
   classifier_out=keras.layers.Flatten()(out)
   classifier_out=keras.layers.Dense(units = 256,activation='sigmoid')(classifier_out)
   classifier_out=keras.layers.BatchNormalization()(classifier_out)
   #classifier_out_1=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_1)
   #classifier_out_1=keras.layers.BatchNormalization()(classifier_out_1)
   classifier_out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out)
   self.model = keras.models.Model( inputs=[layer_input], outputs=[classifier_out])
   self.model.compile( optimizer=keras.optimizers.Adam(lr=0.000001), loss=['categorical_crossentropy'], metrics=['accuracy'])






   
   

