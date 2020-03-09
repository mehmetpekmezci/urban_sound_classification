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

   self.model.summary()



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
  
  loss_weights=[]
  
  if  is_all_data_labeled(y_data_1) :
     loss_weights[0] = 1/5 ## y_data_1 , label
     loss_weights[1] = 1/5 ## y_data_2 , label
     loss_weights[2] = 1/5 ## x_data_1 , autoencoder
     loss_weights[3] = 1/5 ## x_data_2 , autoencoder
     loss_weights[4] = 1/5 ## discriminator
  else :
     loss_weights[0] = 0/5 ## y_data_1 , label
     loss_weights[1] = 0/5 ## y_data_2 , label
     loss_weights[2] = 1/5 ## x_data_1 , autoencoder
     loss_weights[3] = 1/5 ## x_data_2 , autoencoder
     loss_weights[4] = 3/5 ## discriminator
     
  return x_data_1,y_data_one_hot_encoded_1,x_data_2,y_data_one_hot_encoded_2,similarity,loss_weights


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data_1,x_data_2,y_data_1,y_data_2,similarity,loss_weights=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 

  #self.model.fit([x_data_1,x_data_2,loss_weights,y_data_1,y_data_2], [y_data_1,y_data_2,x_data_1,x_data_2,similarity], epochs = 1, batch_size = self.uscData.mini_batch_size,verbose=0)
  self.model.train_on_batch([x_data_1,x_data_2,loss_weights,y_data_1,y_data_2,similarity], y=None, batch_size = self.uscData.mini_batch_size,verbose=0)
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  evaluation = self.model.evaluate(x_data, y_data, batch_size = self.uscData.mini_batch_size,verbose=0)
  trainingLoss = evaluation[0]
  trainingAccuracy = evaluation[1]
  #print(self.model.metrics_names) 
  #print(evaluation) 
  self.trainCount+=1
  if self.trainCount % 100 == 0 :
     self.save_weights()

  return trainingTime,trainingAccuracy,trainingLoss,prepareDataTime
     
 def test(self,data):
  testTimeStart = int(round(time.time())) 
  augment=False
  x_data_1,x_data_2,y_data_1,y_data_2,similarity,loss_weights=self.prepareData(data,augment) 
  evaluation = self.model.evaluate([x_data_1,x_data_2,loss_weights,y_data_1,y_data_2,similarity], [y_data_1,y_data_2,x_data_1,x_data_2,similarity],batch_size = self.uscData.mini_batch_size,verbose=0)
  testAccuracy = evaluation[1]
  testTimeStop = int(round(time.time())) 
  testTime=testTimeStop-testTimeStart
  return testTime,testAccuracy
  
# def buildModel(self):
#   layer_input = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.number_of_time_slices,self.uscData.latent_space_presentation_data_length))
#   out=keras.layers.LSTM(self.lstm_size,dropout=0.2,recurrent_dropout=0.2)(layer_input)
#   out=keras.layers.BatchNormalization()(out)
#   out=keras.layers.Dense(units = 2048,activation='relu')(out)
#   out=keras.layers.BatchNormalization()(out)
#   out=keras.layers.Dropout(0.2)(out)
#   out=keras.layers.Dense(units = 2048,activation='relu')(out)
#   out=keras.layers.BatchNormalization()(out)
#   out=keras.layers.Dropout(0.2)(out)
#   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)
#   self.model = keras.models.Model(inputs=[layer_input], outputs=[out])
#   selectedOptimizer=keras.optimizers.Adam(lr=0.0001)
#   self.model.compile(optimizer=selectedOptimizer, loss='categorical_crossentropy',metrics=['accuracy'])



 def buildModel(self):
   layer_input_loss_weights = keras.layers.Input(shape=(5,))
   layer_input_1 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1))
   layer_input_2 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.track_length,1))
   layer_input_target_label_1 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.number_of_classes,1))
   layer_input_target_label_2 = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,self.uscData.number_of_classes,1))
   layer_input_similarity = keras.layers.Input(batch_shape=(self.uscData.mini_batch_size,1,1))
   layer_input=keras.layers.concatenate([layer_input_1,layer_input_2])
   # Convolution1D(filters, kernel_size,...)
   out=keras.layers.Convolution1D(16, 64,strides=25,activation='relu', padding='same')(layer_input)
   out=keras.layers.Convolution1D(32,32,strides=7,activation='relu', padding='same')(out)
   out=keras.layers.Convolution1D(64, 16,strides=7,activation='relu', padding='same')(out)
   out=keras.layers.BatchNormalization()(out)
   out=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(out)
   out=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(out)
   out=keras.layers.Convolution1D(16, 4,strides=2,activation='relu', padding='same')(out)
   out=keras.layers.BatchNormalization()(out)
   
   common_out=out
   classifier_cnn_out_1=keras.layers.Lambda(lambda x: x[:,0,:,:], name='slice')(common_out)
   classifier_cnn_out_2=keras.layers.Lambda(lambda x: x[:,1,:,:], name='slice')(common_out)
   
   classifier_out_1=keras.layers.Flatten()(classifier_cnn_out_1)
   classifier_out_1=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_1)
   classifier_out_1=keras.layers.BatchNormalization()(classifier_out_1)
   classifier_out_1=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_1)
   classifier_out_1=keras.layers.BatchNormalization()(classifier_out_1)
   classifier_out_1=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out_1)

   classifier_out_2=keras.layers.Flatten()(classifier_cnn_out_2)
   classifier_out_2=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_2)
   classifier_out_2=keras.layers.BatchNormalization()(classifier_out_2)
   classifier_out_2=keras.layers.Dense(units = 128,activation='sigmoid')(classifier_out_2)
   classifier_out_2=keras.layers.BatchNormalization()(classifier_out_2)
   classifier_out_2=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(classifier_out_2)
   
   autoencoder_common_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_out)
   autoencoder_common_out=keras.layers.UpSampling1D(2)(autoencoder_common_out)
   autoencoder_common_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_out)
   autoencoder_common_out=keras.layers.UpSampling1D(2)(autoencoder_common_out)
   autoencoder_common_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_out)
   autoencoder_common_out=keras.layers.UpSampling1D(2)(autoencoder_common_out)
   autoencoder_common_out=keras.layers.BatchNormalization()(autoencoder_common_out)
   autoencoder_common_out=keras.layers.Convolution1D(64,16, activation='relu', padding='same')(common_out)
   autoencoder_common_out=keras.layers.UpSampling1D(7)(autoencoder_common_out)
   autoencoder_common_out=keras.layers.Convolution1D(32,32, activation='relu', padding='same')(common_out)
   autoencoder_common_out=keras.layers.UpSampling1D(7)(autoencoder_common_out)
   autoencoder_common_out=keras.layers.Convolution1D(16,64, activation='relu', padding='same')(common_out)
   autoencoder_common_out=keras.layers.UpSampling1D(25)(autoencoder_common_out)
   autoencoder_common_out=keras.layers.BatchNormalization()(autoencoder_common_out)

   autoencoder_common_out=np.swapaxes(autoencoder_common_out,0,1)
   autoencoder_common_cnn_out_1=autoencoder_common_out[0]
   autoencoder_common_cnn_out_2=autoencoder_common_out[1]

   
   autoencoder_out_1=keras.layers.Flatten()(autoencoder_common_cnn_out_1)
   autoencoder_out_1=keras.layers.Dense(units = 128,activation='sigmoid')(autoencoder_out_1)
   autoencoder_out_1=keras.layers.BatchNormalization()(autoencoder_out_1)

   autoencoder_out_2=keras.layers.Flatten()(autoencoder_out_2)
   autoencoder_out_2=keras.layers.Dense(units = 128,activation='sigmoid')(autoencoder_out_2)
   autoencoder_out_2=keras.layers.BatchNormalization()(autoencoder_out_2)

   discriminator_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(common_out)
   discriminator_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(discriminator_out)
   discriminator_out=keras.layers.Convolution1D(16,4, activation='relu', padding='same')(discriminator_out)
   discriminator_out=keras.layers.Flatten()(discriminator_out)
   discriminator_out=keras.layers.Dense(units = 128,activation='sigmoid')(discriminator_out)
   discriminator_out=keras.layers.BatchNormalization()(discriminator_out)
   discriminator_out=keras.layers.Dense(units = 1,activation='softmax')(discriminator_out)
   
   loss=layer_input_loss_weights[0] * keras.losses.categorical_crossentropy(layer_input_target_label_1,classifier_out_1) + \
        layer_input_loss_weights[1] * keras.losses.categorical_crossentropy(layer_input_target_label_2,classifier_out_2) + \
        layer_input_loss_weights[2] * keras.losses.binary_crossentropy(layer_input_1,autoencoder_out_1) + \
        layer_input_loss_weights[3] * keras.losses.binary_crossentropy(layer_input_2,autoencoder_out_2) + \
        layer_input_loss_weights[4] * keras.losses.binary_crossentropy(layer_input_similarity,discriminator_out) 
   
    
   self.model = keras.models.Model(inputs=[layer_input_1,layer_input_2,layer_input_loss_weights,layer_input_target_label_1,layer_input_target_label_2,layer_input_similarity], outputs=[classifier_out_1,classifier_out_2,autoencoder_out_1,autoencoder_out_2,discriminator_out])
   self.model.add_loss(loss)
   
   selectedOptimizer=keras.optimizers.Adam(lr=0.0001)
   self.model.compile(optimizer=selectedOptimizer, loss=None,metrics=['accuracy'])


