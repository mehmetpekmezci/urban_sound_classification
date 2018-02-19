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

np.set_printoptions(threshold=np.nan)
script_dir=os.path.dirname(os.path.realpath(__file__))
main_data_dir = script_dir+'/../data/'
raw_data_dir = main_data_dir+'/0.raw/UrbanSound8K/audio'
csv_data_dir=main_data_dir+"/1.csv"
fold_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
#fold_dirs = ['fold1']
fold_data_dictionary=dict()


# 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
SOUND_RECORD_SAMPLING_RATE=22050

# 88200 = (2^3)=8 x ( 3^2)=9 x (5^2)=25 x (7^2)=49
#PARALLEL_CONVOLUTION_KERNELS_SIZES=np.array([ [8,9] , [8*25,49],[8*25,63],[9*25,49*4], [8*9,5*49] ])
#PARALLEL_CONVOLUTION_KERNELS_SIZES=np.array([ [2,3,25] , [3,5,49],[5,7,63],[7,2,9], [8,9,49] ])
#PARALLEL_CONVOLUTION_KERNELS_SIZES=np.array([ [7*5,7,7,3,3,2,3,3,2,3,2]  ])
PARALLEL_CONVOLUTION_KERNELS_SIZES=np.array([ [7*5,5,7,3,3,2,3,1,2,1,2,1,2,1]  ])
NUMBER_OF_KERNELS=128
LEARNING_RATE = 0.00001
TRAINING_ITERATIONS = 2000 
MINI_BATCH_SIZE=10
NUMBER_OF_CLASSES=10
NUMBER_OF_FULLY_CONNECTED_NEURONS=512
DROP_OUT=0.5
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
  # NUMBER_OF_KERNELS tane conv_kernel_length_x * conv_kernel_length_y lik convolution kernel uygulanacak , sonucta 1x22050x64 luk tensor cikacak.
  
  #concatanation_of_parallel_conv_layers = tf.placeholder(tf.float32, shape=[None,None])
  concatanation_of_parallel_conv_layers = tf.zeros([MINI_BATCH_SIZE,1],tf.float32)
  #concatanation_of_parallel_conv_layers = tf.zeros([1,1],tf.float32)
                                                              

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
     #conv_stride=1

     with tf.name_scope(convName):
       W_conv = weight_variable_4d([conv_kernel_length_x, conv_kernel_length_y, conv_input_channel, conv_kernel_count])
       b_conv = bias_variable([conv_kernel_count])
       #Based on conv2d doc:
       #    shape of input = [batch, in_height, in_width, in_channels]
       #    shape of filter = [filter_height, filter_width, in_channels, out_channels]
       #    Last dimension of input and third dimension of filter represents the number of input channels. In your case they are not equal.
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
    
  # Fully connected layer 1 -- after 3 round of downsampling, our 1x4*SOUND_RECORD_SAMPLING_RATE (1x4*22050) wav 
  # is down to 4*SOUND_RECORD_SAMPLING_RATE/(pool1_length*pool2_length*pool3_length) (147) feature maps -- maps this to NUMBER_OF_FULLY_CONNECTED_NEURONS features.
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

    # Layer 2 with BN, using Tensorflows built-in BN function
    #w2_BN = tf.Variable(w2_initial)
    #z2_BN = tf.matmul(l1_BN,w2_BN)
    #batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
    #scale2 = tf.Variable(tf.ones([100]))
    #beta2 = tf.Variable(tf.zeros([100]))
    #BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)
    #l2_BN = tf.nn.sigmoid(BN2)




  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print("h_fc1_drop.shape="+str(h_fc1_drop.shape))

  # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS(1024) features to NUMBER_OF_CLASSES(10) classes, one for each class
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable_2d([NUMBER_OF_FULLY_CONNECTED_NEURONS, NUMBER_OF_CLASSES])
    b_fc2 = bias_variable([NUMBER_OF_CLASSES])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print("y_conv.shape="+str(y_conv.shape))
  return y_conv, keep_prob


def conv2d(x, W,stride):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, 1, 1], padding='SAME')


def max_pool_1xL(x,L):
  """max_pool_1x10 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 1, L, 1],
                        strides=[1, 1, L, 1], padding='SAME')

def get_truncated_normal_generator():
    mean=0
    standard_deviation=0.1
    a=-5*standard_deviation
    b=5*standard_deviation
    return truncnorm(a, b, loc=mean, scale=standard_deviation)


def weight_variable_2d(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
#  initial = tf.Print(initial, [initial], message="This is initial: ",summarize=200, first_n=70)
  return tf.Variable(initial)


def weight_variable_4d(shape):
  """weight_variable generates a weight variable of a given shape."""
  

  truncated_normal_generator=get_truncated_normal_generator()
  conv_kernel_length_x=shape[0]
  conv_kernel_length_y=shape[1]
  conv_input_channel=shape[2]
  conv_kernel_count=shape[3]


  initial=np.zeros([conv_kernel_length_x,conv_kernel_length_y,conv_input_channel,conv_kernel_count])

  for i in range(conv_kernel_length_x):
    for j in range(conv_input_channel) :
      for k in range(conv_kernel_count) :
         initial[i,0:conv_kernel_length_y,j,k]=truncated_normal_generator.rvs(conv_kernel_length_y)

  for i in range(conv_kernel_count):
      initialization_type=random.randint(0, 3)
      if initialization_type == 0 :
       # first conv_kernel_length_y/2 elements  are set to 0
       initial[:,:,0:math.floor(conv_kernel_count/2),i]=0
      elif initialization_type == 1 :
       # last conv_kernel_length_y/2 elements  are set to 0
       initial[:,:,math.floor(conv_input_channel/2):,i]=0
      elif initialization_type == 2 :
       ## bir 0 bir 1
       for j in range(conv_input_channel):
             if (j+i)%2==0 :
                initial[:,:,j,i]=0 
      else :
       ## do nothing
       initial=initial
  #print(initial)
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(tf.convert_to_tensor(initial, np.float32))


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def augment_speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    result=np.zeros(len(sound_array))
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    result_calculated= sound_array[ indices.astype(int) ]
    if len(result) > len(result_calculated) :
       result[:len(result_calculated)]=result_calculated
    else :
        result=result_calculated[:len(result)]
    return result

def augment_inverse(sound_array):
    return -sound_array

def augment_translate(snd_array, n):
    """ Translates the sound wave by n indices, fill the first n elements of the array with zeros """
    new_array=np.zeros(len(snd_array))
    for i in range(snd_array.shape[0]):
       if i+n == len(new_array) :
          break
       new_array[i+n]=snd_array[i] 
    return new_array


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
              train_x_data=augment_random(train_data[:,:4*SOUND_RECORD_SAMPLING_RATE])
              train_y_data=train_data[:,4*SOUND_RECORD_SAMPLING_RATE]
              train_y_data_one_hot_encoded=one_hot_encode_array(train_y_data)
              train_step.run(feed_dict={x: train_x_data, y: train_y_data_one_hot_encoded, keep_prob: DROP_OUT})
              trainingTimeStop = int(round(time.time())) 
              totalTime=totalTime+(trainingTimeStop-trainingTimeStart) 
            train_accuracy = accuracy.eval(feed_dict={x: train_x_data, y:train_y_data_one_hot_encoded, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('Total time spent in the fold %s is %g seconds' % (fold, totalTime))
            sys.stdout.flush()

def augment_random(x_data):

  augmented_data= np.zeros([x_data.shape[0],x_data.shape[1]],np.float32)
  for i in range(x_data.shape[0]) :
    augmentation_type=random.randint(0, 9)
    if augmentation_type == 0 :
       SPEED_FACTOR=1.1
       augmented_data[i]= augment_speedx(x_data[i],SPEED_FACTOR)
    elif augmentation_type == 1 :
       SPEED_FACTOR=0.9
       augmented_data[i]= augment_speedx(x_data[i],SPEED_FACTOR)
    elif augmentation_type == 2 :
       SPEED_FACTOR=0.8
       augmented_data[i]= augment_speedx(x_data[i],SPEED_FACTOR)
    elif augmentation_type == 3 :
       SPEED_FACTOR=1.3
       augmented_data[i]= augment_speedx(x_data[i],SPEED_FACTOR)
    elif augmentation_type == 4 :
       SPEED_FACTOR=1.2
       augmented_data[i]= augment_speedx(x_data[i],SPEED_FACTOR)
    elif augmentation_type == 5 :
       TRANSLATION_FACTOR=8820
       augmented_data[i]= augment_translate(x_data[i],TRANSLATION_FACTOR)
    elif augmentation_type == 6 :
       TRANSLATION_FACTOR=4410
       augmented_data[i]= augment_translate(x_data[i],TRANSLATION_FACTOR)
    elif augmentation_type == 6 :
       TRANSLATION_FACTOR=2050
       augmented_data[i]= augment_translate(x_data[i],TRANSLATION_FACTOR)
    elif augmentation_type == 7 :
       augmented_data[i]= -x_data[i]    
    else :
       augmented_data[i]=x_data[i] 
  return augmented_data
 

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


    