#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *

##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class USCModel :
 def __init__(self, uscLogger,uscData,uscCochlea): 
   self.uscLogger             = uscLogger
   self.uscData               = uscData
   self.uscCochlea            = uscCochlea
   ## self.uscData.time_slice_length = 440


   self.uscLogger.logger.info("###########################################")
   self.uscLogger.logger.info("STARTING CLASSIFIER DEFINITION")
   self.uscLogger.logger.info("")
   
   script_dir=os.path.dirname(os.path.realpath(__file__))
   script_name=os.path.basename(script_dir)
   self.model_save_dir=script_dir+"/../../save/"+script_name
   self.model_save_file="model.h5"

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
  if augment==True :
    x_data=self.uscData.augment_random(x_data)
  x_data=self.uscData.normalize(x_data)
  x_data=self.uscCochlea.get_cochlea_mfcc(x_data)
  y_data=data[:,4*self.uscData.sound_record_sampling_rate]
  y_data_one_hot_encoded=self.uscData.one_hot_encode_array(y_data)

  x_data=x_data.reshape(x_data.shape[0],x_data.shape[1],x_data.shape[2],x_data.shape[3],1)
  #self.uscLogger.logger.info("x_data.shape="+str(x_data.shape))

  return x_data,y_data_one_hot_encoded


 def train(self,data,categorical_weight):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 

  #plt.plot(x_data[0,0])
  #plt.show()
  #self.uscData.play(x_data_1[0])

  #self.uscLogger.logger.info("model.train_on_batch started")
  self.model.train_on_batch([x_data],[y_data])
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
  
  self.trainCount+=1
  if self.trainCount % 100 == 0 :
     self.save_weights()

  return trainingTime,trainingLoss ,  trainingAccuracy ,prepareDataTime
 
# def setPredictedLabel(self,data,categorical_weight):
#     x_data_1,x_data_2,y_data_1,y_data_2,similarity=self.prepareData(data,False) 
#     y_pred=self.model.predict([x_data_1,x_data_2,categorical_weight])
#     # we know that similarity is real label even for youtube data, so we will use it to augment accuracy 
#     if data[0][0]> 0 : # cheap randomness 50%, (may be the other prediction is true :) )
#         predicted_value=y_pred[0]+y_pred[1]*similarity
#         ## how to use similarity ?:
#         ## if similar :
#         ##   if y_pred_0 != y_pred_1 :
#         ##        wrong prediction , then use arg_max(y_pred_0+y_pred_1)
#         ##   else :
#         ##        at least the prediction obeys the similarity, so again use argmax(y_pred_0+y_pred_1)
#         ## if not similar:
#         ##   use y_pred_0
#     else :
#         predicted_value=y_pred[1]+y_pred[0]*similarity
#
#     data[:,4*self.uscData.sound_record_sampling_rate]= np.argmax(predicted_value, axis=1)
#     return data

 def test(self,data,categorical_weight):
  testTimeStart = int(round(time.time())) 
  augment=False
  prepareDataTimeStart = int(round(time.time())) 
  x_data,y_data=self.prepareData(data,augment) 
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  
  
  evaluation = self.model.evaluate([x_data], [y_dat],batch_size = self.uscData.mini_batch_size,verbose=0)
  y_pred=self.model.predict([x_data])
  y_pred= np.argmax(y_pred[0], axis=1)
  y_raw_data= np.argmax(y_data, axis=1)
  confusionMatrix=tf.math.confusion_matrix(labels=y_raw_data, predictions=y_pred,num_classes=self.uscData.number_of_classes).numpy()
 
  
  testLoss = evaluation[0]
  testAccuracy = evaluation[1]
  
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  
  return testTime,testLoss ,testAccuracy ,prepareDataTime,confusionMatrix
  

 def buildModel(self):

   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.number_of_windows,self.uscData.mfcc_image_dimensions[0],self.uscData.mfcc_image_dimensions[1],1),name="layer_input")
   self.uscLogger.logger.info("layer_input.shape="+str(layer_input.shape))

   #out=keras.layers.TimeDistributed(keras.layers.Convolution2D(64, (4, 2),activation='relu', padding='valid'), input_shape=(self.uscData.number_of_windows, self.uscData.mfcc_image_dimensions[0],self.uscData.mfcc_image_dimensions[1]))(layer_input) 
   out=keras.layers.TimeDistributed(keras.layers.Convolution2D(64,(4,2),activation='relu'), input_shape=(self.uscData.number_of_windows, self.uscData.mfcc_image_dimensions[0],self.uscData.mfcc_image_dimensions[1],1))(layer_input) 
   self.uscLogger.logger.info("1. out.shape="+str(out.shape))

   out=keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2,1),strides=(2,1)))(out) 
   self.uscLogger.logger.info("2. out.shape="+str(out.shape))
   
   out=keras.layers.TimeDistributed(keras.layers.Convolution2D(64, (4, 2),activation='relu'))(out) 
   self.uscLogger.logger.info("3. out.shape="+str(out.shape))

   out=keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 1),strides=(2,1)))(out) 
   self.uscLogger.logger.info("4. out.shape="+str(out.shape))

   out=keras.layers.TimeDistributed(keras.layers.Convolution2D(64, (4, 2),activation='relu'))(out) 
   self.uscLogger.logger.info("5. out.shape="+str(out.shape))

   out=keras.layers.TimeDistributed(keras.layers.Flatten())(out)
   self.uscLogger.logger.info("6. out.shape="+str(out.shape))

   out=keras.layers.Dropout(0.5)(out)
   out=keras.layers.LSTM(128, return_sequences=False, dropout=0.5)(out)
   self.uscLogger.logger.info("7. out.shape="+str(out.shape))


   out=keras.layers.Dense(units = 64,activation='sigmoid')(out)
   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)
   self.model = keras.models.Model(inputs=[layer_input],outputs=[out])
   self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss=['categorical_crossentropy'],metrics=[['accuracy']])






   
   

