#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *
from USCModel import *


def main(_):
  uscLogger=USCLogger()
  uscData=USCData(uscLogger.logger)
  uscData.prepareData()
  with tf.Session() as session:
   session.run(tf.global_variables_initializer())
   uscModel=USCModel(session,uscLogger,uscData)
   for trainingIterationNo in range(uscModel.training_iterations):
    uscLogger.logStepStart(session,trainingIterationNo)
    prepareDataTimes=[ ]
    trainingTimes=[ ]
    trainingAccuracies=[ ]
    testTimes=[ ]
    testAccuracies=[ ]
    for fold in np.random.permutation(uscData.fold_dirs):
       #uscLogger.logger.info("  Fold : "+str(fold))
       current_fold_data=uscData.get_fold_data(fold)
       for current_batch_counter in range(int(current_fold_data.shape[0]/uscModel.mini_batch_size)) :
         if (current_batch_counter+1)*uscModel.mini_batch_size <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*uscModel.mini_batch_size:(current_batch_counter+1)*uscModel.mini_batch_size,:]
         else:
           batch_data=current_fold_data[current_batch_counter*uscModel.mini_batch_size:,:]
         if fold == "fold10":
             ## FOLD10 is reserved for testing
              #uscLogger.logger.info("  Testing Batch Counter : "+str(current_batch_counter))
              testTime,testAccuracy=uscModel.test(batch_data)
              testTimes.append(testTime)
              testAccuracies.append(testAccuracy)
         else:
              #uscLogger.logger.info("  Training Batch Counter : "+str(current_batch_counter))
              trainingTime,trainingAccuracy,prepareDataTime=uscModel.train(batch_data)
              trainingTimes.append(trainingTime)
              trainingAccuracies.append(trainingAccuracy)
              prepareDataTimes.append(prepareDataTime)
    uscLogger.logStepEnd(session,prepareDataTimes,trainingTimes,trainingAccuracies,testTimes,testAccuracies,trainingIterationNo)

if __name__ == '__main__':
 tf.app.run(main=main)

