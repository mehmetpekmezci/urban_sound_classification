#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *





##
## NueralNetworkModel will be as :
## CNN LAYERS + LSTM LAYERS + FULLY CONNECTED LAYER + SOFTMAX
##
class USCAutoEncoder :
 def __init__(self, session, uscLogger,uscData): 
   self.session               = session
   self.uscLogger             = uscLogger
   self.uscData               = uscData
   self.time_window_size      = 10
   script_dir=os.path.dirname(os.path.realpath(__file__))
   script_name=os.path.basename(script_dir)
   self.model_save_dir=script_dir+"/../../save/"+script_name
   self.model_save_file="autoEndoer.h5"
   ## self.uscData.time_slice_length = 440
   ## so we will have nearly 400 time steps in 4 secs record. (88200) (with %50 overlapping)
   ## so we again sliced the input data into 20 (num_of_paral_lstms)
   ## each lstm cell works on one part only (for example lstm[0] works on the begginning of the data
   ## (self.num_of_paralel_lstms  = 28) x (self.lstm_time_steps = 18) = (self.uscData.number_of_time_slices=504)
   self.training_iterations   = 10
   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True
   self.buildModel()
   self.load_weights()
   self.trainCount=0

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
  x_data=self.uscData.overlapping_slice(x_data)
  ## returns -> (batch_size, number_of_time_slices, time_slice_length)
  #x_data=self.uscData.fft(x_data)
  x_data_list = self.uscData.convert_to_list_of_word2vec_window_sized_data(x_data)
  ## returns -> list of (mini_batch_size,word2vec_window_size,time_slice_lentgh), this list has self.number_of_time_slices/self.word2vec_window_size elements
  return x_data_list


 def train(self,data):
  augment=True
  prepareDataTimeStart = int(round(time.time())) 
  x_data_list=self.prepareData(data,augment)
  prepareDataTimeStop = int(round(time.time())) 
  prepareDataTime=prepareDataTimeStop-prepareDataTimeStart
  trainingTimeStart = int(round(time.time())) 
  for x_data in x_data_list :
   ## x_data of size (mini_batch_size,word2vec_window_size,time_slice_lentgh)
  ## pull the data in the middle
   #y_data=np.concatenate((x_data[:,:int(len(x_data)/2+1),:],x_data[:,int(len(x_data)/2+1):,:]))
   y_data=np.delete(x_data,int(x_data.shape[1]/2),1)
   y_data=y_data.reshape(y_data.shape[0],y_data.shape[1]*y_data.shape[2],1)
   x_data=x_data[:,int(x_data.shape[1]/2),:].reshape(x_data.shape[0],x_data.shape[2],1)
   self.optimizer.run(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob:0.5})
 
  trainingTimeStop = int(round(time.time())) 
  trainingTime=trainingTimeStop-trainingTimeStart
  trainingLossTotal=0
  for x_data in x_data_list :
   y_data=np.delete(x_data,int(x_data.shape[1]/2),1)
   y_data=y_data.reshape(y_data.shape[0],y_data.shape[1]*y_data.shape[2],1)
   x_data=x_data[:,int(x_data.shape[1]/2),:].reshape(x_data.shape[0],x_data.shape[2],1)
   evaluation = self.accuracy.eval(feed_dict={self.x_input: x_data, self.real_y_values:y_data, self.keep_prob: 1.0})
   #trainingLossTotal+=evaluation[0]
   trainingLossTotal+=evaluation
  trainingLoss=trainingLossTotal/len(x_data_list)
  #print(self.model.metrics_names) 
  #print(evaluation) 
  self.trainCount+=1
  
  if self.trainCount % 100 :
     self.save_weights()
  
  return trainingTime,trainingLoss,prepareDataTime

       
 def buildCNNLayer(inputLayer,filterSize,kernelSize,strideSize,poolSize,activation,upsampleFactor,residuals=False):
     cnnInputChannel=int(inputLayer.shape[3])
     cnnOutputChannel=filterSize
     cnnKernelSizeX=1
     cnnKernelSizeY=kernelSize
     cnnStrideSizeX  = 1 
     cnnStrideSizeY  = strideSize                     
     cnnPoolSizeX    = 1
     cnnPoolSizeY    = poolSize
     
     W = tf.Variable(tf.truncated_normal([cnnKernelSizeX, cnnKernelSizeY, cnnInputChannel, cnnOutputChannel], stddev=0.1))
     B = tf.Variable(tf.constant(0.1, shape=[cnnOutputChannel]))
     C = tf.nn.conv2d(inputLayer,W,strides=[1,cnnStrideSizeX, cnnStrideSizeY, 1], padding='SAME')+B     
     ## APPLY ACTIVATION FN
     H=C  
     if activation is not None :
       H = tf.nn.relu(C)

     ## APPLY RESIDUALS
     R=H 
     if residuals and cnnInputChannel > 1 :
       if cnnInputChannel!=cnnOutputChannel :
         WR = tf.Variable(tf.truncated_normal([1, 1, int(H.shape[3]), cnnInputChannel], stddev=0.1))
         R=tf.nn.conv2d(H,WR,strides=[1,1, 1, 1], padding='SAME')
       R=tf.concat((R,inputLayer),2)
       ## while doubling output, to diminish the output size we appply max pooling
       R = tf.nn.max_pool(R, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') 

     ## APPLY POOLING
     P=R
     if poolSize is not None :
       P = tf.nn.max_pool(R, ksize=[1, cnnPoolSizeX,cnnPoolSizeY, 1],strides=[1, cnnPoolSizeX,cnnPoolSizeY , 1], padding='SAME') 
     
     ## UPSAMPLE
     U=P
     if upsampleSize is not None:
         W1 = tf.Variable(tf.truncated_normal([1, 1,int(P.shape[3]), int(P.shape[3]) * upsampleFactor ], stddev=0.1))
         U=tf.nn.conv2d(inputLayer,W1,strides=[1,1, 1, 1], padding='SAME')

     return U



     
 def buildModel(self):

   self.real_y_values = tf.placeholder(tf.float32, shape=(self.mini_batch_size, self.output_size), name="real_y_values")
 
   ## x_data
   self.model_input=tf.placeholder(tf.float32, shape=(self.uscData.mini_batch_size, self.uscData.time_slice_length,1))

   ## buildCNNLayer(inputLayer,filterSize,kernelSize,strideSize,poolSize,activation,upsampleFactor,residuals=False)
   
   
   ## fourier cnn layers
   e=buildCNNLayer(self.model_input,64,3,1,4,None,None)
   e=buildCNNLayer(e,64,3,1,4,None,None)
   e=buildCNNLayer(e,64,3,1,4,None,None)

   ## normal cnn layers   
   e=buildCNNLayer(e,32,3,1,None,'relu',None)
   e=buildCNNLayer(e,32,3,1,None,'relu',None)
   e=buildCNNLayer(e,16,3,1,None,'relu',None)
   e=buildCNNLayer(e,16,3,1,None,'relu',None)
   e=buildCNNLayer(e,8,3,1,None,'relu',None)

   self.encoder=e
   self.uscLogger.logger.info("shape of encoder = "+str(self.encoder.shape))
   self.uscData.latent_space_presentation_data_length=int(self.encoder.shape[1])
   
   ae=buildCNNLayer(e,8,3,1,None,'relu',2)
   e=buildCNNLayer(e,16,3,1,None,'relu',None)
   e=buildCNNLayer(e,16,3,1,None,'relu',2)
   e=buildCNNLayer(e,32,3,1,None,'relu',None)
   e=buildCNNLayer(e,32,3,1,None,'relu',None)

   ## fourier cnn layers
   e=buildCNNLayer(model_input,64,3,1,None,None,4)
   e=buildCNNLayer(e,64,3,1,None,None,4)
   e=buildCNNLayer(e,64,3,1,None,None,4)


   # output is of size (word2vec_window_size-1) like skipgram
   e=buildCNNLayer(e,int(5*(self.uscData.word2vec_window_size-1)),3,1,None,None,None)
   self.uscLogger.logger.info("shape of model_output = "+str( e.shape))
   
   
   e_flat = tf.reshape( e, [-1, int(e.shape[1]*e.shape[2])] )
   self.logger.info("e_flat="+str( e_flat))


   self.loss=tf.losses.mean_squared_error(self.real_y_values,self.model_output)
   self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
   


#   out=keras.layers.Dense(units = self.uscData.number_of_classes,activation='softmax')(out)
#   model = keras.models.Model(inputs=inputs, outputs=[out])

 def encode(self,x_data):
   encodeTimeStart = int(round(time.time()))
   x_data_swapped=np.swapaxes(x_data,0,1)
   x_data_list=[]
   for i in range(x_data_swapped.shape[0]):
       x_data_list.append(x_data_swapped[i])
   x_encoded_data_list=[]
   ##  (self.mini_batch_size      ,self.number_of_time_slices,self.time_slice_length)  to
   ##  (self.number_of_time_slices,self.mini_batch_size      ,self.time_slice_length)
   for x_data_item in x_data_list :
       #self.uscLogger.logger.info("len(x_data_item)="+str( len(x_data_item)))
       x_data_item=x_data_item.reshape(x_data_item.shape[0],x_data_item.shape[1],1)
       encoded_x_data_item=self.encoder.predict(x_data_item)
       #self.uscLogger.logger.info("encoded_x_data_item.shape="+str( encoded_x_data_item.shape))
       x_encoded_data_list.append(encoded_x_data_item)
   encoded_x_data=np.asarray(x_encoded_data_list)
   encodedValue=np.swapaxes(encoded_x_data,0,1)
   #self.uscLogger.logger.info("encodedValue.shape="+str( encodedValue.shape))
   encodeTimeStop = int(round(time.time()))
   encodeTime=encodeTimeStop-encodeTimeStart
   return encodedValue,encodeTime    






