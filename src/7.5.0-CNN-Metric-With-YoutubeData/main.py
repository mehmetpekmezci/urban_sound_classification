#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *
from USCModel import *


def main(_):
 uscLogger=USCLogger()
 uscData=USCData(uscLogger.logger)
 uscData.prepareData()
 uscData.findListOfYoutubeDataFiles()
  
 config = tf.ConfigProto()
 config.gpu_options.allow_growth = True

 with tf.Session(config=config) as session:
   session.run(tf.global_variables_initializer())

   uscModel=USCModel(session,uscLogger,uscData)
   for trainingIterationNo in range(uscModel.training_iterations):
    uscLogger.logStepStart(session,trainingIterationNo)
    prepareDataTimes=[ ]
    trainingTimes=[ ]
    trainingAccuracies=[ ]
    trainingLosses=[]
    testTimes=[ ]
    testAccuracies=[ ]
    ## youtube dta also contain fold1-9 of urban sound data 
    current_youtube_data_as_list=np.random.permutation(uscData.loadNextYoutubeData())
    for current_batch_counter in range(math.floor(len(current_youtube_data_as_list)/uscData.mini_batch_size)) :
         if current_batch_counter % 100 == 0:
           uscLogger.logger.info("Training Batch No : "+str(current_batch_counter))
         batch_data=current_youtube_data_as_list[current_batch_counter*uscData.mini_batch_size:(current_batch_counter+1)*uscData.mini_batch_size]
         #uscLogger.logger.info("batch_data.shape: "+str(batch_data.shape))
         trainingTime,trainingAccuracy,trainingLoss,prepareDataTime=uscModel.train(batch_data)
         trainingTimes.append(trainingTime)
         trainingAccuracies.append(trainingAccuracy)
         prepareDataTimes.append(prepareDataTime)
         trainingLosses.append(trainingLoss)
    uscLogger.logger.info("Testing with fol10 of urban sound data... ")
    ## FOLD10 is reserved for testing
    current_fold_data=np.random.permutation(uscData.get_fold_data("fold10"))
    for current_batch_counter in range(int(current_fold_data.shape[0]/uscData.mini_batch_size)) :
         if (current_batch_counter+1)*uscData.mini_batch_size <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*uscData.mini_batch_size:(current_batch_counter+1)*uscData.mini_batch_size,:]
         else:
           batch_data=current_fold_data[current_batch_counter*uscData.mini_batch_size:,:]
         #uscLogger.logger.info("  Testing Batch Counter : "+str(current_batch_counter))
         testTime,testAccuracy=uscModel.test(batch_data)
         testTimes.append(testTime)
         testAccuracies.append(testAccuracy)
    uscLogger.logStepEnd(session,prepareDataTimes,trainingTimes,trainingAccuracies,trainingLosses,testTimes,testAccuracies,trainingIterationNo)

if __name__ == '__main__':
 tf.app.run(main=main)

