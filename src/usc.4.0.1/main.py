#!/usr/bin/env python3
from header import *
from data import *
from lstm import *

def main(_):
  prepareData()
  load_all_csv_data_back_to_memory()
  normalize_all_data()
  



  ## prepare input variable place holder. First layer is input layer with 1xSOUND_DATA_LENGTH  matrix. (in other words vector with length SOUND_DATA_LENGTH, , or in other words 1x1xSOUND_DATA_LENGTH tensor )
  x = tf.placeholder(tf.float32, shape=[None,4*SOUND_RECORD_SAMPLING_RATE])
  ## prepare output variable place holder, one hot encoded 
  y = tf.placeholder(tf.float32, shape=[None,NUMBER_OF_CLASSES])

  with tf.Session() as sess:
#    test_x_data_sliced=slice_data(get_fold_data('fold1'),NUMBER_OF_TIME_SLICES)
#    print(test_x_data_sliced.shape)

#    test_x_data_reshaped=tf.reshape(get_fold_data('fold1'),[-1,NUMBER_OF_TIME_SLICES,-1])
#    print(test_x_data_reshaped.shape)

#    exit(0)

    lstm_net = ModelNetwork(in_size = LSTM_INPUT_SIZE,lstm_size = LSTM_SIZE,num_layers = LSTM_NUMBER_OF_LAYERS,out_size = LSTM_OUTPUT_SIZE,session = sess,learning_rate = 0.00001,name = "urban_sound_rnn_network")
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
              print(test_x_data.shape)
              test_x_data_sliced=slice_data(test_x_data,NUMBER_OF_TIME_SLICES)
              print(test_x_data_sliced.shape)
              test_y_data=test_data[:,4*SOUND_RECORD_SAMPLING_RATE]
              test_y_data_one_hot_encoded=one_hot_encode_array(test_y_data)
              test_predicted_y, test_accuracy = lstm_net.test_batch(test_x_data_sliced,test_y_data_one_hot_encoded)
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
            # MP asagidaki for dongusunde +1 olunca hatali tensor uretiyor tensorflow exception atiyor.
            #for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)+1) :
            for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
              trainingTimeStart = int(round(time.time()))  
              if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
                  train_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
              else:
                  train_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
              train_x_data=train_data[:,:4*SOUND_RECORD_SAMPLING_RATE]
              train_x_data_sliced=slice_data(train_x_data,NUMBER_OF_TIME_SLICES)
              train_y_data=train_data[:,4*SOUND_RECORD_SAMPLING_RATE]
              train_y_data_one_hot_encoded=one_hot_encode_array(train_y_data)
              train_accuracy = lstm_net.train_batch(train_x_data_sliced, train_y_data_one_hot_encoded)
              trainingTimeStop = int(round(time.time())) 
              totalTime=totalTime+(trainingTimeStop-trainingTimeStart) 
            
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print('Total time spent in the fold %s is %g seconds' % (fold, totalTime))
            sys.stdout.flush()
 

if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,
                     default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





  
