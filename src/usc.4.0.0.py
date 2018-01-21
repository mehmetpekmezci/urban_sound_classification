#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:46:52 2017
@author: mpekmezci
"""
from scipy.stats import truncnorm
import math
import tensorflow as tf
import urllib3
import tarfile
import csv
import glob
import sys
import os
import argparse
import sys
import tempfile
import numpy as np
import librosa
import pandas as pd
import time
import random
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

script_dir=os.path.dirname(os.path.realpath(__file__))
main_data_dir = script_dir+'/../data/'
raw_data_dir = main_data_dir+'/0.raw/UrbanSound8K/audio'
csv_data_dir=main_data_dir+"/1.csv"
fold_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
fold_data_dictionary=dict()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

net = ModelNetwork(in_size = in_size,lstm_size = lstm_size,num_layers = num_layers,
		   out_size = out_size,session = sess,learning_rate = 0.0001,
		   name = "human_position_rnn_network")

# 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
NUMBER_OF_SLICES=22050
SOUND_RECORD_SAMPLING_RATE=22050
NUMBER_OF_CLASSES=10
MAX_VALUE_FOR_NORMALIZATION=0
MIN_VALUE_FOR_NORMALIZATION=0



def parse_audio_files():
    global raw_data_dir,csv_data_dir,fold_dirs
    sub4SecondSoundFilesCount=0
    for sub_dir in fold_dirs:
      print("Parsing : "+sub_dir)
      csvDataFile=open(csv_data_dir+"/"+sub_dir+".csv", 'w')
      csvDataWriter = csv.writer(csvDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      for file_path in glob.glob(os.path.join(raw_data_dir, sub_dir, '*.wav')):
         print(file_path)
         try :
          classNumber=file_path.split('/')[-1].split('.')[0].split('-')[1]
          sound_data,sampling_rate=librosa.load(file_path)
          sound_data=np.array(sound_data)
          sound_data_duration=int(sound_data.shape[0]/SOUND_RECORD_SAMPLING_RATE)
          if sound_data_duration < 4 :
             sub4SecondSoundFilesCount=sub4SecondSoundFilesCount+1
             sound_data_in_4_second=np.zeros(4*SOUND_RECORD_SAMPLING_RATE)
             for i in range(sound_data.shape[0]):
               sound_data_in_4_second[i]=sound_data[i]
          else  :  
             sound_data_in_4_second=sound_data[:4*SOUND_RECORD_SAMPLING_RATE]
          sound_data_in_4_second=np.append(sound_data_in_4_second,[classNumber])
          csvDataWriter.writerow(sound_data_in_4_second)       
         except :
                e = sys.exc_info()[0]
                print ("Exception :")
                print (e)
      csvDataFile.close()       
    print("sub4SecondSoundFilesCount="+str(sub4SecondSoundFilesCount));  


def prepareData():
    print ("prepareData function ...")
    if not  os.path.exists(raw_data_dir) :
       if not  os.path.exists(main_data_dir+'/../data/0.raw'):
         os.makedirs(main_data_dir+'/../data/0.raw')   
    if not os.path.exists(main_data_dir+"/0.raw/UrbanSound8K"):
      if os.path.exists(main_data_dir+"/0.raw/UrbanSound8K.tar.gz"):
         print("Extracting "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar = tarfile.open(main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar.extractall(main_data_dir+'/../data/0.raw')
         tar.close()
         print("Extracted "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
      else :   
        print("download "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz from http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2  using firefox browser or chromium  and re-run this script")
         #print("download "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz from https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound8k.html using firefox browser or chromium  and re-run this script")
        exit(1)
#         http = urllib3.PoolManager()
#         chunk_size=100000
#         r = http.request('GET', 'http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2', preload_content=False)
#         with open(main_data_dir+"/0.raw/UrbanSound8K.tar.gz", 'wb') as out:
#          while True:
#           data = r.read(chunk_size)
#           if not data:
#             break
#           out.write(data)
#         r.release_conn()

    if not os.path.exists(csv_data_dir) :
       os.makedirs(csv_data_dir)  
       parse_audio_files()
    print ("prepareData function finished ...")

def normalize(data):
    global MAX_VALUE_FOR_NORMALIZATION , MIN_VALUE_FOR_NORMALIZATION
    data_normalized=(data-MIN_VALUE_FOR_NORMALIZATION)/(MAX_VALUE_FOR_NORMALIZATION-MIN_VALUE_FOR_NORMALIZATION)
    return data_normalized

def one_hot_encode_array(arrayOfYData):
   returnMatrix=np.empty([0,NUMBER_OF_CLASSES]);
   for i in range(arrayOfYData.shape[0]):
        one_hot_encoded_class_number = np.zeros(NUMBER_OF_CLASSES)
        one_hot_encoded_class_number[int(arrayOfYData[i])]=1
        returnMatrix=np.row_stack([returnMatrix, one_hot_encoded_class_number])
   return returnMatrix

def one_hot_encode(classNumber):
   one_hot_encoded_class_number = np.zeros(NUMBER_OF_CLASSES)
   one_hot_encoded_class_number[int(classNumber)]=1
   return one_hot_encoded_class_number

def deepnn(x):
  epsilon = 1e-3
  with tf.name_scope('reshape'):
    print("x.shape="+str(x.shape))
    x_image = tf.reshape(x, [-1, 1, int(x.shape[1]), 1])

  ## for all convolutions our kernel is one dimensional
  conv_kernel_length_x=1
  conv_kernel_count=NUMBER_OF_KERNELS 
  
  concatanation_of_parallel_conv_layers = tf.zeros([MINI_BATCH_SIZE,1],tf.float32)

  for columnNo in range(PARALLEL_CONVOLUTION_KERNELS_SIZES.shape[0]) :

   conv_input_channel=1
   previous_level_conv_output=x_image

   for depth in range(PARALLEL_CONVOLUTION_KERNELS_SIZES.shape[1]) :

     convSize=PARALLEL_CONVOLUTION_KERNELS_SIZES[columnNo][depth]

     convName="conv_size_"+str(convSize)+"_depth_"+str(depth)
     conv_kernel_length_y=convSize
     conv_stride=math.ceil(convSize/3)
     if conv_stride < 2 :
        conv_stride=2

     with tf.name_scope(convName):
       W_conv = weight_variable_4d([conv_kernel_length_x, conv_kernel_length_y, conv_input_channel, conv_kernel_count])
       b_conv = bias_variable([conv_kernel_count])
       h_conv = tf.nn.relu(conv2d(previous_level_conv_output, W_conv,conv_stride) + b_conv)
       print(convName+"_h.shape="+str(h_conv.shape))

     with tf.name_scope(convName+"_pool"):
       # Max Pooling layer - downsamples by pool_length.
       pool_length=conv_stride
       h_pool = max_pool_1xL(h_conv,pool_length)
       print(convName+"_h_pool.shape="+str(h_pool.shape))
    
       ## put the output of this layer to the next depth's input layer.
       previous_level_conv_output=h_pool
     conv_input_channel=conv_kernel_count

   with tf.name_scope("conv_size_"+str(convSize)+"_flatten"):
    previous_level_conv_output_flat=tf.contrib.layers.flatten(previous_level_conv_output)
    #previous_level_conv_output_flat=tf.squeeze(previous_level_conv_output)
    print("conv_size_"+str(convSize)+"_flatten.shape="+str( previous_level_conv_output_flat.shape))

   with tf.name_scope("conv_size_"+str(convSize)+"_concat"):
    print(concatanation_of_parallel_conv_layers)
    print(previous_level_conv_output_flat)
    concatanation_of_parallel_conv_layers=tf.concat([concatanation_of_parallel_conv_layers,previous_level_conv_output_flat],1)
    print(concatanation_of_parallel_conv_layers)
    print("concatanation_of_parallel_conv_layers.shape="+str( concatanation_of_parallel_conv_layers.shape))
    
  with tf.name_scope('fc1'):
    print(concatanation_of_parallel_conv_layers.shape)
    W_fc1 = weight_variable_2d([int(concatanation_of_parallel_conv_layers.shape[1]), NUMBER_OF_FULLY_CONNECTED_NEURONS])
    print("W_fc1.shape="+str(W_fc1.shape))
    b_fc1 = bias_variable([NUMBER_OF_FULLY_CONNECTED_NEURONS])
    print("b_fc1.shape="+str(b_fc1.shape))
    matmul_fc1=tf.matmul(concatanation_of_parallel_conv_layers, W_fc1)+b_fc1
    print("matmul_fc1.shape="+str(matmul_fc1.shape))
    batch_mean, batch_var = tf.nn.moments(matmul_fc1,[0])
    scale = tf.Variable(tf.ones(NUMBER_OF_FULLY_CONNECTED_NEURONS))
    beta = tf.Variable(tf.zeros(NUMBER_OF_FULLY_CONNECTED_NEURONS))
    batch_normalization_fc1 = tf.nn.batch_normalization(matmul_fc1,batch_mean,batch_var,beta,scale,epsilon)
    print("batch_normalization_fc1.shape="+str(batch_normalization_fc1.shape))
    h_fc1 = tf.nn.relu( batch_normalization_fc1 )
    print("h_fc1.shape="+str(h_fc1.shape))

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print("h_fc1_drop.shape="+str(h_fc1_drop.shape))

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable_2d([NUMBER_OF_FULLY_CONNECTED_NEURONS, NUMBER_OF_CLASSES])
    b_fc2 = bias_variable([NUMBER_OF_CLASSES])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print("y_conv.shape="+str(y_conv.shape))
  return y_conv, keep_prob


def main(_):
  prepareData()
  load_all_csv_data_back_to_memory()
  normalize_all_data()

  ## prepare input variable place holder. First layer is input layer with 1xSOUND_DATA_LENGTH  matrix. (in other words vector with length SOUND_DATA_LENGTH, , or in other words 1x1xSOUND_DATA_LENGTH tensor )
  x = tf.placeholder(tf.float32, shape=[None,4*SOUND_RECORD_SAMPLING_RATE])
    
  ## prepare output variable place holder, one hot encoded 
  y = tf.placeholder(tf.float32, shape=[None,NUMBER_OF_CLASSES])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv)

  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(TRAINING_ITERATIONS):
        for fold in np.random.permutation(fold_dirs):
          if fold == "fold10":
            total_test_accuracy=0
            number_of_tests=0
            current_fold_data=get_fold_data(fold)
            for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
              if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
                test_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
              else:
                test_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
              test_x_data=test_data[:,:4*SOUND_RECORD_SAMPLING_RATE]
              test_y_data=test_data[:,4*SOUND_RECORD_SAMPLING_RATE]
              test_y_data_one_hot_encoded=one_hot_encode_array(test_y_data)
              test_accuracy = accuracy.eval(feed_dict={x: test_x_data, y:test_y_data_one_hot_encoded, keep_prob: 1.0})
              number_of_tests=number_of_tests+1
              total_test_accuracy=total_test_accuracy+test_accuracy 
              print('test accuracy %g' % (test_accuracy))
            print('Mean Test Accuracy %g' % (total_test_accuracy/number_of_tests))
          else:
            totalTime=0.0  
            print('Started training for fold : '+fold)

            loadTimeStart = int(round(time.time()))  
            current_fold_data=get_fold_data(fold)
            loadTimeStop = int(round(time.time())) 
            print('Total time spent loading the fold %s is %g seconds' % (fold, (loadTimeStop-loadTimeStart)))
            print(current_fold_data.shape)
            # MP asagidaki for dongusunde +1 olunca hatali tensor uretiyor tensorflow exception atiyor.
            #for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)+1) :
            for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
              trainingTimeStart = int(round(time.time()))  
              if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
                  train_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
              else:
                  train_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
              train_x_data=train_data[:,:4*SOUND_RECORD_SAMPLING_RATE]
              train_y_data=train_data[:,4*SOUND_RECORD_SAMPLING_RATE]
              train_y_data_one_hot_encoded=one_hot_encode_array(train_y_data)
              train_step.run(feed_dict={x: train_x_data, y: train_y_data_one_hot_encoded, keep_prob: DROP_OUT})
              trainingTimeStop = int(round(time.time())) 
              totalTime=totalTime+(trainingTimeStop-trainingTimeStart) 
            train_accuracy = accuracy.eval(feed_dict={x: train_x_data, y:train_y_data_one_hot_encoded, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('Total time spent in the fold %s is %g seconds' % (fold, totalTime))
            sys.stdout.flush()


def load_all_csv_data_back_to_memory():
     print ("load_all_csv_data_back_to_memory function started ...")
     global fold_data_dictionary, MAX_VALUE_FOR_NORMALIZATION ,  MIN_VALUE_FOR_NORMALIZATION
     for fold in fold_dirs:
       fold_data_dictionary[fold]=np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
       for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*SOUND_RECORD_SAMPLING_RATE]
          maxOfArray=np.amax(loadedDataX)
          minOfArray=np.amin(loadedDataX)
          if MAX_VALUE_FOR_NORMALIZATION < maxOfArray :
              MAX_VALUE_FOR_NORMALIZATION = maxOfArray
          if MIN_VALUE_FOR_NORMALIZATION > minOfArray :
              MIN_VALUE_FOR_NORMALIZATION = minOfArray
          ## Then append Y data to the end of row
          fold_data_dictionary[fold][i]=np.append(loadedDataX,loadedDataY)
     print ("load_all_csv_data_back_to_memory function finished ...")

def normalize_all_data():
     print ("normalize_all_data function started ...")
     global fold_data_dictionary
     for fold in fold_dirs:
       for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*SOUND_RECORD_SAMPLING_RATE]
          normalizedLoadedDataX=normalize(loadedDataX)
          fold_data_dictionary[fold][i]=np.append(normalizedLoadedDataX,loadedDataY)
     print ("normalize_all_data function finished ...")

def get_fold_data(fold):
     global fold_data_dictionary
     return np.random.permutation(fold_data_dictionary[fold])

if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,
                     default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





class ModelNetwork:
	def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn"):
		self.scope = name

		self.in_size = in_size
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.out_size = out_size

		self.session = session

		self.learning_rate = tf.constant( learning_rate )

		# Last state of LSTM, used when running the network in TEST mode
		self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

		with tf.variable_scope(self.scope):
			## (batch_size, timesteps, in_size)
			self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
			self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

			# LSTM
			self.lstm_cells = [ tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
			self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

			# Iteratively compute output of recurrent network
			outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)

			# Linear activation (FC layer on top of the LSTM net)
			self.rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=0.01 ))
			self.rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=0.01 ))

			outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
			network_output = ( tf.matmul( outputs_reshaped, self.rnn_out_W ) + self.rnn_out_B )

			batch_time_shape = tf.shape(outputs)
			#self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )
			self.final_outputs = tf.reshape( network_output, (batch_time_shape[0], batch_time_shape[1], self.out_size) )


			## Training: provide target outputs for supervised training.
			self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
			y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

			#self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long) )
			self.cost = tf.losses.mean_squared_error(y_batch_long,network_output) 

			self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

			#self.train_op  = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)




	## Input: X is a single element, not a list!
	def run_step(self, x, init_zero_state=True):
		## Reset the initial state of the network.
		if init_zero_state:
			init_value = np.zeros((self.num_layers*2*self.lstm_size,))
		else:
			init_value = self.lstm_last_state

		out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xinput:[x], self.lstm_init_value:[init_value]   } )

		self.lstm_last_state = next_lstm_state[0]

		return out[0][0]


	## xbatch must be (batch_size, timesteps, input_size)
	## ybatch must be (batch_size, timesteps, output_size)
	def train_batch(self, xbatch, ybatch):
		init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))

		cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.xinput:xbatch, self.y_batch:ybatch, self.lstm_init_value:init_value   } )

		return cost

test_action_no=10
test_subject_no=5
test_example_no=2
TEST_FRAME = data_dictionary[test_action_no][test_subject_no][test_example_no][0]
TOTAL_ACTIONS=20
TOTAL_SUBJECTS=10
TOTAL_EXAMPLE=3
NUMBER_OF_TRAINING_STEPS= 100 * TOTAL_ACTIONS * TOTAL_SUBJECTS * TOTAL_EXAMPLE
in_size = out_size = FRAME_DATA_POINT_COUNT * 3 # 20 data points times 3 (X,Y,Z)
lstm_size = 256
num_layers = 2
NUMBER_OF_TEST_FRAMES = 25 # Number of test human position frames to generate after training the network
time_steps = 100


## Initialize the network


sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())




      for i in range(len(data_dictionary[action_no][subject_no][example_no])) :
        input_frame=data_dictionary[action_no][subject_no][example_no][i]
        input_frame=np.reshape(input_frame[:,0:3], (60))
        if i == len(data_dictionary[action_no][subject_no][example_no])-1 :
          output_frame=data_dictionary[action_no][subject_no][example_no][0]
        else :
          output_frame=data_dictionary[action_no][subject_no][example_no][i+1]

        output_frame=np.reshape(output_frame[:,0:3], (60))
        batch[:, i, :] = input_frame
        batch_y[:, i, :] = output_frame

      cst = net.train_batch(batch, batch_y)
      if trainingStep%100 == 0  :   
        print ("Training Time = ",str(time.time()-start_time),"  Training Cost = ", cst, "  Training Step =",trainingStep)
        start_time=time.time()
     


 saver.save(sess, "saved.model.ckpt")





## 2) GENERATE NUMBER_OF_TEST_FRAMES FRAMES USING THE TRAINED NETWORK

if os.path.isfile(ckpt_file) :
	saver.restore(sess, ckpt_file)

generated_frames=[]
generated_frames.append(TEST_FRAME)
TEST_FRAME_INPUT=np.reshape(TEST_FRAME[:,0:3], (1,60))

for i in range(NUMBER_OF_TEST_FRAMES):
    print(i)
    input_=TEST_FRAME_INPUT
    output_ = net.run_step( input_ , i==0)
   # output_=np.reshape(output_, (60))
    print(output_.shape)
    TEST_FRAME_INPUT=np.reshape(output_, (1,60))
    generated_frames.append(np.reshape(output_, (20,3)))
    


print(len(generated_frames))
print(generated_frames)
animate(generated_frames,"generated_frames")







