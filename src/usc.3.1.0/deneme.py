#!/usr/bin/env python3
from header import *
from data import *
from NeuralNetworkModel import *


def main(_):

  logger.info("Main is started ... ")
  # load all data into the memory

  load_all_csv_data_back_to_memory()
  data=get_fold_data('fold1')
  TRACK_LENGTH=20
  NUMBER_OF_TIME_SLICES=4
  SLIDE_STEP=2
  x_data=data[:2,:20]
  x_data_reshaped3=np.reshape(x_data,(x_data.shape[0],NUMBER_OF_TIME_SLICES,int(TRACK_LENGTH/NUMBER_OF_TIME_SLICES),1))
  x_data_reshaped=np.zeros((2,NUMBER_OF_TIME_SLICES,int(TRACK_LENGTH/NUMBER_OF_TIME_SLICES+(NUMBER_OF_TIME_SLICES-1)*SLIDE_STEP),1))
  x_data_reshaped2=np.zeros((2,NUMBER_OF_TIME_SLICES,int(TRACK_LENGTH/NUMBER_OF_TIME_SLICES+(NUMBER_OF_TIME_SLICES-1)*SLIDE_STEP),1))
  
  print(time.time())
  for i in range(2) :
   for j in range(NUMBER_OF_TIME_SLICES) :
    for k in range(int(TRACK_LENGTH/NUMBER_OF_TIME_SLICES)) :
      t=x_data[i,int(j*TRACK_LENGTH/NUMBER_OF_TIME_SLICES+k)]
      x_data_reshaped[i,j,k+j*SLIDE_STEP,0]=t

  print(time.time())
  for j in range(NUMBER_OF_TIME_SLICES) :
      x_data_reshaped2[:,j,int(j*SLIDE_STEP):int(j*SLIDE_STEP+TRACK_LENGTH/NUMBER_OF_TIME_SLICES),:]=x_data_reshaped3[:,j,:,:]
  
  print(time.time())

  print(x_data)
#  print(x_data_reshaped)
  print(x_data_reshaped2)
#  print(x_data_reshaped3)


if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,
                     default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





  
