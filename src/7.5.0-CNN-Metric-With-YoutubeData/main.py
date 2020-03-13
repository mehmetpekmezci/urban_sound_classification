#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *
from USCModel import *



def main(_):
 uscLogger=USCLogger()
 uscData=USCData(uscLogger.logger)
 uscModel=USCModel(uscLogger,uscData)

 uscData.prepareData()
 uscData.findListOfYoutubeDataFiles()
  

 for trainingIterationNo in range(uscModel.training_iterations):
   
    
    logs=dict()
    
    logs['trainingPrepareDataTimes']=[]
    logs['trainingTimes']=[]
    logs['trainingAccuracies_classifier_1']=[]
    logs['trainingAccuracies_classifier_2']=[]
    logs['trainingAccuracies_autoencoder_1']=[]
    logs['trainingAccuracies_autoencoder_2']=[]
    logs['trainingAccuracies_discriminator']=[]
    logs['trainingLosses_total']=[]
    logs['trainingLosses_classifier_1']=[]
    logs['trainingLosses_classifier_2']=[]
    logs['trainingLosses_autoencoder_1']=[]
    logs['trainingLosses_autoencoder_2']=[]
    logs['trainingLosses_discriminator']=[]
    
    logs['testPrepareDataTimes']=[]
    logs['testTimes']=[]
    logs['testAccuracies_classifier_1']=[]
    logs['testAccuracies_classifier_2']=[]
    logs['testAccuracies_autoencoder_1']=[]
    logs['testAccuracies_autoencoder_2']=[]
    logs['testAccuracies_discriminator']=[]
    logs['testLosses_total']=[]
    logs['testLosses_classifier_1']=[]
    logs['testLosses_classifier_2']=[]
    logs['testLosses_autoencoder_1']=[]
    logs['testLosses_autoencoder_2']=[]
    logs['testLosses_discriminator']=[]
    
 
    ## youtube dta also contain fold1-9 of urban sound data 
    current_youtube_data_as_list=np.random.permutation(uscData.loadNextYoutubeData())
    
    uscLogger.logStepStart(trainingIterationNo)
     
     
    for current_batch_counter in range(math.floor(len(current_youtube_data_as_list)/uscData.mini_batch_size)) :
    

           
         batch_data=current_youtube_data_as_list[current_batch_counter*uscData.mini_batch_size:(current_batch_counter+1)*uscData.mini_batch_size]
         
         #uscLogger.logger.info("batch_data.shape: "+str(batch_data.shape))
         trainingTime,trainingLoss ,  trainingLoss_classifier_1 ,  trainingLoss_classifier_2 ,  trainingLoss_autoencoder_1 ,  trainingLoss_autoencoder_2 ,  trainingLoss_discriminator ,  trainingAccuracy_classifier_1 ,  trainingAccuracy_classifier_2 ,  trainingAccuracy_autoencoder_1 ,  trainingAccuracy_autoencoder_2 ,  trainingAccuracy_discriminator ,prepareDataTime = uscModel.train(batch_data)
         
         logs['trainingPrepareDataTimes'].append(prepareDataTime)
         logs['trainingTimes'].append(trainingTime)
         logs['trainingAccuracies_classifier_1'].append(trainingAccuracy_classifier_1)
         logs['trainingAccuracies_classifier_2'].append(trainingAccuracy_classifier_2)
         logs['trainingAccuracies_autoencoder_1'].append(trainingAccuracy_autoencoder_1)
         logs['trainingAccuracies_autoencoder_2'].append(trainingAccuracy_autoencoder_2)
         logs['trainingAccuracies_discriminator'].append(trainingAccuracy_discriminator)
         logs['trainingLosses_total'].append(trainingLoss)
         logs['trainingLosses_classifier_1'].append(trainingLoss_classifier_1)
         logs['trainingLosses_classifier_2'].append(trainingLoss_classifier_2)
         logs['trainingLosses_autoencoder_1'].append(trainingLoss_autoencoder_1)
         logs['trainingLosses_autoencoder_2'].append(trainingLoss_autoencoder_2)
         logs['trainingLosses_discriminator'].append(trainingLoss_discriminator)
         
         if current_batch_counter % 100 == 0:
           uscLogger.logStepEnd('YoutubeData'+str(current_batch_counter),logs,trainingIterationNo)
         
         

    for fold in np.random.permutation(uscData.fold_dirs):

       current_fold_data=np.random.permutation(uscData.get_fold_data(fold))
       for current_batch_counter in range(int(current_fold_data.shape[0]/uscData.mini_batch_size)) :
         if (current_batch_counter+1)*uscData.mini_batch_size <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*uscData.mini_batch_size:(current_batch_counter+1)*uscData.mini_batch_size,:]
         else:
           batch_data=current_fold_data[current_batch_counter*uscData.mini_batch_size:,:]
         if fold == "fold10":
             ## FOLD10 is reserved for testing
              #uscLogger.logger.info("  Testing Batch Counter : "+str(current_batch_counter))
              testTime,testLoss ,  testLoss_classifier_1 ,  testLoss_classifier_2 ,  testLoss_autoencoder_1 ,  testLoss_autoencoder_2 ,  testLoss_discriminator ,  testAccuracy_classifier_1 ,  testAccuracy_classifier_2 ,  testAccuracy_autoencoder_1 ,  testAccuracy_autoencoder_2 ,  testAccuracy_discriminator ,prepareDataTime=uscModel.test(batch_data)
              logs['testPrepareDataTimes'].append(prepareDataTime)
              logs['testTimes'].append(testTime)
              logs['testAccuracies_classifier_1'].append(testAccuracy_classifier_1)
              logs['testAccuracies_classifier_2'].append(testAccuracy_classifier_2)
              logs['testAccuracies_autoencoder_1'].append(testAccuracy_autoencoder_1)
              logs['testAccuracies_autoencoder_2'].append(testAccuracy_autoencoder_2)
              logs['testAccuracies_discriminator'].append(testAccuracy_discriminator)
              logs['testLosses_total'].append(testLoss)
              logs['testLosses_classifier_1'].append(testLoss_classifier_1)
              logs['testLosses_classifier_2'].append(testLoss_classifier_2)
              logs['testLosses_autoencoder_1'].append(testLoss_autoencoder_1)
              logs['testLosses_autoencoder_2'].append(testLoss_autoencoder_2)
              logs['testLosses_discriminator'].append(testLoss_discriminator)
         else:
              #uscLogger.logger.info("  Training Batch Counter : "+str(current_batch_counter))
              trainingTime,trainingLoss ,  trainingLoss_classifier_1 ,  trainingLoss_classifier_2 ,  trainingLoss_autoencoder_1 ,  trainingLoss_autoencoder_2 ,  trainingLoss_discriminator ,  trainingAccuracy_classifier_1 ,  trainingAccuracy_classifier_2 ,  trainingAccuracy_autoencoder_1 ,  trainingAccuracy_autoencoder_2 ,  trainingAccuracy_discriminator ,prepareDataTime = uscModel.train(batch_data)


              logs['trainingPrepareDataTimes'].append(prepareDataTime)
              logs['trainingTimes'].append(trainingTime)
              logs['trainingAccuracies_classifier_1'].append(trainingAccuracy_classifier_1)
              logs['trainingAccuracies_classifier_2'].append(trainingAccuracy_classifier_2)
              logs['trainingAccuracies_autoencoder_1'].append(trainingAccuracy_autoencoder_1)
              logs['trainingAccuracies_autoencoder_2'].append(trainingAccuracy_autoencoder_2)
              logs['trainingAccuracies_discriminator'].append(trainingAccuracy_discriminator)
              logs['trainingLosses_total'].append(trainingLoss)
              logs['trainingLosses_classifier_1'].append(trainingLoss_classifier_1)
              logs['trainingLosses_classifier_2'].append(trainingLoss_classifier_2)
              logs['trainingLosses_autoencoder_1'].append(trainingLoss_autoencoder_1)
              logs['trainingLosses_autoencoder_2'].append(trainingLoss_autoencoder_2)
              logs['trainingLosses_discriminator'].append(trainingLoss_discriminator)
       uscLogger.logStepEnd(fold,logs,trainingIterationNo)

    uscLogger.logStepEnd('Epoch END',logs,trainingIterationNo)

if __name__ == '__main__':
 tf.compat.v1.app.run(main=main)

