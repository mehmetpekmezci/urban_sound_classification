#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:46:52 2017
@author: mpekmezci
"""
import tensorflow as tf
#import urllib3
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
#fold_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
fold_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
fold_data_dictionary=dict()


LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 40
SOUND_RECORD_SAMPLING_RATE=22050
MINI_BATCH_SIZE=5
NUMBER_OF_CLASSES=10
NUMBER_OF_FULLY_CONNECTED_NEURONS=1024
LOW_MEMORY_MODE=0

def parse_audio_files():
    ## each wav file in the folds are 22050 Sampling rate and 4 seconds length.
    ## The wav files are split into 22050 part (1 second). Then inserted into corresponding csv file as a new data point (a row)
    
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
    if not  os.path.exists(raw_data_dir) :
       if not  os.path.exists(main_data_dir+'/../data/0.raw'):
         os.makedirs(main_data_dir+'/../data/0.raw')   
       if os.path.exists(main_data_dir+"/0.raw/UrbanSound8K.tar.gz") :
         #print("Extracting "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar = tarfile.open(main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar.extractall(main_data_dir+'/../data/0.raw')
         tar.close()
       else :   
         print("download "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz from https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound8k.html by hand and re-run this script")
         exit(1)
#     http = urllib3.PoolManager()
#     chunk_size=100000
#     r = http.request('GET', 'https://www.google.com/url?q=http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id%3D2&sa=D&ust=1504002907380000&usg=AFQjCNHVd9935Q5lVS5SRtBjuuwEZrJR8w', preload_content=False)
#     with open(main_data_dir+"/0.raw/UrbanSound8K.tar.gz", 'wb') as out:
#      while True:
#         data = r.read(chunk_size)
#         if not data:
#             break
#         out.write(data)
#     r.release_conn()

    if not os.path.exists(csv_data_dir) :
       os.makedirs(csv_data_dir)  
       parse_audio_files()
       augment_data()
       shuffle_csv_files()
       normalize_csv_files()

def augment_data():
    global csv_data_dir,fold_dirs
    for fold in fold_dirs:
      print("Augmenting Data in the Fold :"+fold)
      csvAugmentedDataFile=open(csv_data_dir+"/"+fold+".augmented.csv", 'w')
      csvAugmenteDataWriter = csv.writer(csvAugmentedDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      csvData=np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
      for i in range(csvData.shape[0]) :

          csvDataLine=csvData[i]

          ## first write data itself 
          csvAugmenteDataWriter.writerow(csvDataLine)
         
          ## fold10 is test fold so no need to augment
          if fold == "fold10" :
            continue 
        
          csvDataLineX=csvDataLine[:4*SOUND_RECORD_SAMPLING_RATE]
          csvDataLineY=csvDataLine[4*SOUND_RECORD_SAMPLING_RATE]

          TRANSLATION_FACTOR=2000
          augmentDataX=augment_translate(csvDataLineX,TRANSLATION_FACTOR)
          augmentedDataXY=np.append(augmentDataX,csvDataLineY)
          csvAugmenteDataWriter.writerow(augmentedDataXY)

          SPEED_FACTOR=1.1
          augmentDataX=augment_speedx(csvDataLineX,SPEED_FACTOR)
          if augmentDataX.shape[0] > 4*SOUND_RECORD_SAMPLING_RATE:
              new_array=augmentDataX[:4*SOUND_RECORD_SAMPLING_RATE]
              augmentDataX=new_array;
          if augmentDataX.shape[0] < 4*SOUND_RECORD_SAMPLING_RATE:
               new_array=np.zeros(4*SOUND_RECORD_SAMPLING_RATE)
               new_array[:len(augmentDataX)]=augmentDataX
               augmentDataX=new_array;
          augmentedDataXY=np.append(augmentDataX,csvDataLineY)
          csvAugmenteDataWriter.writerow(augmentedDataXY)

          PITCH_SHIFT_FACTOR=1.1
          augmentDataX=augment_pitchshift(csvDataLineX,PITCH_SHIFT_FACTOR)
          if augmentDataX.shape[0] > 4*SOUND_RECORD_SAMPLING_RATE:
              augmentDataX=augmentDataX[:4*SOUND_RECORD_SAMPLING_RATE]
          if augmentDataX.shape[0] < 4*SOUND_RECORD_SAMPLING_RATE:
               new_array=np.zeros(4*SOUND_RECORD_SAMPLING_RATE)
               new_array[:len(augmentDataX)]=augmentDataX
               augmentDataX=new_array;
          augmentedDataXY=np.append(augmentDataX,csvDataLineY)
          csvAugmenteDataWriter.writerow(augmentedDataXY)

      csvAugmentedDataFile.close()

def shuffle_csv_files():
    global csv_data_dir,fold_dirs
    for fold in fold_dirs:
      print("Shuffling Fold :"+fold)
      csvShuffledDataFile=open(csv_data_dir+"/"+fold+".augmented.shuffled.csv", 'w')
      csvShuffledDataWriter = csv.writer(csvShuffledDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

      MAX_DATA_READ=500
      for df in pd.read_csv(csv_data_dir+"/"+fold+".augmented.csv", chunksize=MAX_DATA_READ,header=None, iterator=True):
        csvData=np.array(df)
        csvShuffledData=np.random.permutation(csvData)
        for i in range(csvShuffledData.shape[0]) :
          shuffledRow=csvShuffledData[i]
          csvShuffledDataWriter.writerow(shuffledRow)
      csvShuffledDataFile.close()
#      os.remove(csv_data_dir+"/"+fold+".augmented.csv")

def normalize_csv_files():
    global csv_data_dir,fold_dirs
    for fold in fold_dirs:
      print("Normalizing Fold :"+fold)
      csvNormalizedShuffledDataFile=open(csv_data_dir+"/"+fold+".augmented.shuffled.normalized.csv", 'w')
      csvNormalizedShuffledDataWriter = csv.writer(csvNormalizedShuffledDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')


      MAX_DATA_READ=500
      for df in pd.read_csv(csv_data_dir+"/"+fold+".augmented.shuffled.csv", chunksize=MAX_DATA_READ,header=None, iterator=True):
        csvShuffledData=np.array(df)
        for i in range(csvShuffledData.shape[0]) :
          shuffledData=csvShuffledData[i]
          shuffledDataX=shuffledData[:4*SOUND_RECORD_SAMPLING_RATE]
          shuffledDataY=shuffledData[4*SOUND_RECORD_SAMPLING_RATE]
          ## only normalize X data
          normalizedShuffledDataX=normalize(shuffledDataX)
          ## Then append Y data to the end of row
          normalizedShuffledDataXY=np.append(normalizedShuffledDataX,shuffledDataY)
          csvNormalizedShuffledDataWriter.writerow(normalizedShuffledDataXY)
      csvNormalizedShuffledDataFile.close()
#      os.remove(csv_data_dir+"/"+fold+".augmented.shuffled.csv")



def augment_speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]



def augment_stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( int(len(sound_array) /f + window_size))

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays

        i=int(i)
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] = result[i2 : i2 + window_size] +  hanning_window*a2_rephased

    result = ((2**(16-4)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')


def augment_pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = augment_stretch(snd_array, 1.0/factor, window_size, h)
    return augment_speedx(stretched[window_size:], factor)


def augment_translate(snd_array, n):
    """ Translates the sound wave by n indices, fill the first n elements of the array with random numbers """
    new_array=np.zeros(len(snd_array))
    for i in range(snd_array.shape[0]):
       if i+n == len(new_array) :
          break
       new_array[i+n]=snd_array[i] 
    return new_array




def normalize(data):
    data_normalized=(data-data.min(0))/data.ptp(0)
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
  with tf.name_scope('reshape'):
    print("x.shape="+str(x.shape))
    x_image = tf.reshape(x, [-1, 1, 4*SOUND_RECORD_SAMPLING_RATE, 1])

  conv1_kernel_length_x=1
  conv1_kernel_length_y=100
  conv1_stride=10
  conv1_input_channel=1
  conv1_kernel_count=64 # 64 tane 1x3 luk convolution kernel uygulanacak , sonucta 1x22050x64 luk tensor cikacak.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([conv1_kernel_length_x, conv1_kernel_length_y, conv1_input_channel, conv1_kernel_count])
    b_conv1 = bias_variable([conv1_kernel_count])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,conv1_stride) + b_conv1)
    print("h_conv1.shape="+str(h_conv1.shape))

  # Pooling layer - downsamples by 2X.
  pool1_length=conv1_stride
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_1xL(h_conv1,pool1_length)
    print("h_pool1.shape="+str(h_pool1.shape))


  # Second convolutional layer -- maps 64 feature maps to 32.
  conv2_kernel_length_x=1
  conv2_kernel_length_y=50
  conv2_stride=10
  conv2_input_channel=conv1_kernel_count
  conv2_kernel_count=64 # 32 tane 1x5 lik convolution kernel uygulanacak , sonucta 1x(22050/pool1_length)x32 lik tensor cikacak.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([conv2_kernel_length_x, conv2_kernel_length_y, conv2_input_channel, conv2_kernel_count])
    b_conv2 = bias_variable([conv2_kernel_count])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,conv2_stride) + b_conv2)
    print("h_conv2.shape="+str(h_conv2.shape))

  # Second pooling layer.
  pool2_length=conv2_stride
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_1xL(h_conv2,pool2_length)
    print("h_pool2.shape="+str(h_pool2.shape))

  # Third convolutional layer -- maps 32 feature maps to 16.
  conv3_kernel_length_x=1
  conv3_kernel_length_y=10
  conv3_stride=6
  conv3_input_channel=conv2_kernel_count
  conv3_kernel_count=64 # 16 tane 1x10 luk convolution kernel uygulanacak , sonucta 1x(22050/pool1_length/pool2_length)x16 lik tensor cikacak.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([conv3_kernel_length_x, conv3_kernel_length_y, conv3_input_channel, conv3_kernel_count])
    b_conv3 = bias_variable([conv3_kernel_count])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,conv3_stride) + b_conv3)
    print("h_conv3.shape="+str(h_conv3.shape))

  # Third pooling layer.
  pool3_length=conv3_stride
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_1xL(h_conv3,pool3_length)
    print("h_pool3.shape="+str(h_pool3.shape))

  # Fully connected layer 1 -- after 3 round of downsampling, our 1x4*SOUND_RECORD_SAMPLING_RATE (1x4*22050) wav 
  # is down to 4*SOUND_RECORD_SAMPLING_RATE/(pool1_length*pool2_length*pool3_length) (147) feature maps -- maps this to NUMBER_OF_FULLY_CONNECTED_NEURONS features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([int(4*SOUND_RECORD_SAMPLING_RATE/(pool1_length*pool2_length*pool3_length)*conv3_kernel_count), NUMBER_OF_FULLY_CONNECTED_NEURONS])
    b_fc1 = bias_variable([NUMBER_OF_FULLY_CONNECTED_NEURONS])

    h_pool3_flat = tf.reshape(h_pool3, [-1,int(4*SOUND_RECORD_SAMPLING_RATE/(pool1_length*pool2_length*pool3_length)*conv3_kernel_count)])
    print("h_pool3_flat.shape="+str(h_pool3_flat.shape))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    print("h_fc1.shape="+str(h_fc1.shape))

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print("h_fc1_drop.shape="+str(h_fc1_drop.shape))

  # Map the NUMBER_OF_FULLY_CONNECTED_NEURONS(1024) features to NUMBER_OF_CLASSES(10) classes, one for each class
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([NUMBER_OF_FULLY_CONNECTED_NEURONS, NUMBER_OF_CLASSES])
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


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def main(_):
  prepareData()

  ## prepare input variable place holder. First layer is input layer with 1xSOUND_DATA_LENGTH  matrix. (in other words vector with length SOUND_DATA_LENGTH, , or in other words 1x1xSOUND_DATA_LENGTH tensor )
  x = tf.placeholder(tf.float32, shape=[None,4*SOUND_RECORD_SAMPLING_RATE])
    
  ## prepare output variable place holder, one hot encoded 
  y = tf.placeholder(tf.float32, shape=[None,NUMBER_OF_CLASSES])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=y_conv)
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
             ## fol10 is test data
             continue;
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
              train_step.run(feed_dict={x: train_x_data, y: train_y_data_one_hot_encoded, keep_prob: 0.5})
              trainingTimeStop = int(round(time.time())) 
              totalTime=totalTime+(trainingTimeStop-trainingTimeStart) 
            train_accuracy = accuracy.eval(feed_dict={x: train_x_data, y:train_y_data_one_hot_encoded, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('Total time spent in the fold %s is %g seconds' % (fold, totalTime))
            sys.stdout.flush()


    fold="fold10"
    #for df in pd.read_csv(csv_data_dir+"/"+fold+".augmented.shuffled.normalized.csv", chunksize=MINI_BATCH_SIZE,header=None, iterator=True):
    for df in pd.read_csv(csv_data_dir+"/"+fold+".csv", chunksize=MINI_BATCH_SIZE,header=None, iterator=True):
       test_data=np.array(df)
       test_x_data=test_data[:,:4*SOUND_RECORD_SAMPLING_RATE]
       test_y_data=test_data[:,4*SOUND_RECORD_SAMPLING_RATE]
       test_y_data_one_hot_encoded=one_hot_encode_array(test_y_data)
       test_accuracy = accuracy.eval(feed_dict={x: test_x_data, y:test_y_data_one_hot_encoded, keep_prob: 1.0})
       print('test accuracy %g' % (test_accuracy))

#    fold="fold10"
#    test_data=get_fold_data(fold)
#    test_x_data=test_data[:,:SOUND_RECORD_SAMPLING_RATE]
#    test_y_data=test_data[:,SOUND_RECORD_SAMPLING_RATE]
#    test_y_data_one_hot_encoded=one_hot_encode_array(test_y_data)
#    test_accuracy = accuracy.eval(feed_dict={x: test_x_data, y:test_y_data_one_hot_encoded, keep_prob: 1.0})
#    print('test accuracy %g' % (test_accuracy))

def get_fold_data(fold):
     global fold_data_dictionary
     if LOW_MEMORY_MODE == 1 :
            print (" LOW_MEMORY_MODE==1 so each fold is loaded every time they called, when the pointer finishes its work, it releases the memory")
            return np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".augmented.shuffled.normalized.csv", "rb"), delimiter=","))
            #return np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
     else:
            if not fold in fold_data_dictionary :
                fold_data_dictionary[fold]=np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".augmented.shuffled.normalized.csv", "rb"), delimiter=","))
                #fold_data_dictionary[fold]=np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
            return fold_data_dictionary[fold]


if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,
                     default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


    
