#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *

##   INPUT  : Any Length of sound data , and sampling_rate
##   OUTPUT : Array of MFCC images ( N, MFCC_IMAGE_DIM_1, MFCC_IMGE_DIM_2)   
##      N--------------------:   ( input_sound_data_size/(window_size*(10^-3)*sampling_rate) )/overlap_ratio  ## 10^-3 is milliseconds
##      MFCC_IMAGE_DIM_1-----:   n_mfcc (20 by default in librosa, we will use probably 40)
##      MFCC_IMAGE_DIM_2-----:   (window_size*(10^-3)*sampling_rate) / MFCC_FFT_WINDOW_SIZE ## MFCC_FFT_WINDOW_SIZE=512 by default
##
##
##

class USCCochlea :
 def __init__(self, uscLogger,uscData): 
   self.uscLogger             = uscLogger
   self.uscData               = uscData

   self.uscLogger.logger.info("###########################################")
   self.uscLogger.logger.info("STARTING COCHLEA DEFINITION")
   self.uscLogger.logger.info("")
   
   script_dir=os.path.dirname(os.path.realpath(__file__))
   script_name=os.path.basename(script_dir)
   self.model_save_dir=script_dir+"/../../save/"+script_name
   self.model_save_file="cochlea.h5"
   
   self.training_iterations = 2000000
   self.buildModel()
   self.load_weights()
   self.trainCount=0
   self.model.summary(print_fn=uscLogger.logger.info)
   self.uscLogger.logger.info("COCHLEA DEFINITION IS FINISHED")
   self.uscLogger.logger.info("###########################################")
   self.uscLogger.logger.info("")


 def load_weights(self):
     if os.path.exists(self.model_save_dir+"/"+self.model_save_file):
         self.model.load_weights(self.model_save_dir+"/"+self.model_save_file)

 def save_weights(self):
     if not os.path.exists(self.model_save_dir):
         os.makedirs(self.model_save_dir)
     self.model.save_weights(self.model_save_dir+"/"+self.model_save_file)


 def prepareData(self,data):
  x_data=data[:,:4*self.uscData.sound_record_sampling_rate]
  x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  mfcc=self.uscData.analytical_mfcc(x_data)
  y_data=mfcc.rehape(mfcc.shape[0],mfcc.shape[1]*mfcc.shape[2]*mfcc.shape[3])
  x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],1)
  return x_data,y_data


 def train(self,data,categorical_weight):
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
 
  #plt.plot(x_data[0])
  #plt.show()
  #self.uscData.play(x_data[0])
  
  self.model.train_on_batch([x_data],[y_data] )

  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart

  #sample_weight
  evaluation = self.model.evaluate([x_data], [y_data],  batch_size = self.uscData.mini_batch_size,verbose=0)
  #prediction = self.model.predict([x_data_1,x_data_2,categorical_weight])

  #self.uscLogger.logger.info(self.model.metrics_names)
  #self.uscLogger.logger.info(evaluation)
  

  trainingLoss = evaluation[0]
  trainingAccuracy = evaluation[1]


  
  self.trainCount+=1
  if self.trainCount % 100 == 0 :
     self.save_weights()

  #return trainingTime,trainingLoss ,  categorical_weight[0][0][0]*trainingLoss_classifier_1 ,  categorical_weight[0][0][0]*trainingLoss_classifier_2 ,  0 ,  0 ,  trainingLoss_discriminator ,  categorical_weight[0][0][0]*trainingAccuracy_classifier_1 ,  categorical_weight[0][0][0]*trainingAccuracy_classifier_2 ,  0 ,  0 ,  trainingAccuracy_discriminator ,prepareDataTime
  return trainingTime,trainingLoss,trainingAccuracy,prepareDataTime
 
 def get_cochlea_mfcc(self,x_data):
     y_pred=self.model.predict([x_data])
     y_pred.reshape(y_pred.shape[0],self.uscData.number_of_windows,self.uscData.mfcc_image_dimensions[0],self.uscData.mfcc_image_dimensions[1])
     return y_pred

 def test(self,data,categorical_weight):
  testTimeStart = int(round(time.time())) 
  augment=False
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment) 
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  
  
  evaluation = self.model.evaluate([x_data], [y_data],batch_size = self.uscData.mini_batch_size,verbose=0)

  testLoss = evaluation[0]
  testAccuracy= evaluation[1]
  
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  
  return testTime,testLoss ,testAccuracy,prepareDataTime
  

 def buildModel(self):

   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1),name="layer_input")
   self.uscLogger.logger.info("layer_input.shape="+str(layer_input.shape))

   out=keras.layers.Convolution1D(64, 64,strides=16,activation='relu', padding='same')(layer_input)
   out=keras.layers.Dropout(0.2)(out)
   
   out=keras.layers.Convolution1D(64, 64,strides=16,activation='relu', padding='same')(layer_input)
   
   out=keras.layers.Convolution1D(64, 64,strides=16,activation='relu', padding='same')(layer_input)
   
   out=keras.layers.Convolution1D(64, 64,strides=16,activation='relu', padding='same')(layer_input)
   
   out=keras.layers.Convolution1D(64, 64,strides=16,activation='relu', padding='same')(layer_input)
      
   out=keras.layers.BatchNormalization()(out)

   out=keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)

   out=keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)

   out=keras.layers.Convolution1D(16, 4,activation='relu', padding='same')(out)
   
   out=keras.layers.Dense(units = self.uscData.number_of_windows*self.uscData.mfcc_image_dimensions[0]*self.uscData.mfcc_image_dimensions[1],activation='sigmoid')(out)
   
   
   
   self.model = keras.models.Model(
                          inputs=[layer_input], 
                          outputs=[out]
                          )
   
   self.model.compile(
       optimizer=keras.optimizers.Adam(lr=0.0001),
       loss=['mse'],
       metrics=[['accuracy']]
   )





   
   

