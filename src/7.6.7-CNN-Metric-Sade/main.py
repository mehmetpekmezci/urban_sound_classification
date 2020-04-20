#!/usr/bin/env python3
from USCHeader import *
from USCLogger import *
from USCData import *
from USCModel import *



def main(_):
 uscLogger=USCLogger()
 uscData=USCData(uscLogger.logger)
 uscModel=USCModel(uscLogger,uscData)

 for trainingIterationNo in range(uscModel.training_iterations):
     
    epoch_logs=uscLogger.getNewLogDictionary()
  

    
    uscLogger.logStepStart(trainingIterationNo)
     
    mode='Training' 
     
    ### YOUTUBE DATA
    #if trainingIterationNo % 5 == 0 and uscLogger.lastAccuracy > 0.6 :
    #if (trainingIterationNo+1) % 20 == 0  :
    if (trainingIterationNo+1) % 100000 == 0  : ## never use youtube data
      while uscData.current_youtube_data is None:
        uscLogger.logger.info('Waiting 5 seconds for youtube data loader thread  ....')
        time.sleep(5)
      current_youtube_data_as_list=uscData.getNextYoutubeData()
    
      for current_batch_counter in range(math.floor(len(current_youtube_data_as_list)/uscData.mini_batch_size)) :
    
         stage_logs=uscLogger.getNewLogDictionary()
         batch_data=np.random.permutation(current_youtube_data_as_list[current_batch_counter*uscData.mini_batch_size:(current_batch_counter+1)*uscData.mini_batch_size])
         
         if uscLogger.lastAccuracy > 1 : ## disable this branch >1           
           batch_data=uscModel.setPredictedLabel(batch_data,np.ones((uscData.mini_batch_size,1,1)))
           logData = uscModel.train(batch_data,np.ones((uscData.mini_batch_size,1,1)))
           uscLogger.appendLogData(stage_logs[mode],logData,False,mode)
           uscLogger.appendLogData(epoch_logs[mode],logData,False,mode)
         else :
           logData = uscModel.train(batch_data,np.zeros((uscData.mini_batch_size,1,1)))
           uscLogger.appendLogData(stage_logs[mode],logData,True,mode)
           uscLogger.appendLogData(epoch_logs[mode],logData,True,mode)
         
         if current_batch_counter % 50 == 0:
           uscLogger.logStepEnd('YoutubeData-'+str(current_batch_counter),mode,stage_logs,trainingIterationNo)
               
    ### YOUTUBE DATA



    for fold in np.random.permutation(uscData.fold_dirs):

       stage_logs=uscLogger.getNewLogDictionary()

       current_fold_data=np.random.permutation(uscData.get_fold_data(fold))
       for current_batch_counter in range(int(current_fold_data.shape[0]/uscData.mini_batch_size)) :
       

    
         if (current_batch_counter+1)*uscData.mini_batch_size <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*uscData.mini_batch_size:(current_batch_counter+1)*uscData.mini_batch_size,:]
         else:
           batch_data=current_fold_data[current_batch_counter*uscData.mini_batch_size:,:]
           
         #self.play(self.augment_echo(x_data[5],2.5))
         plt.plot(batch_data[9]*100)
         plt.show()
         uscData.play(batch_data[9]*100)
         sys.exit(0)         
         
         
         if fold == "fold10":
              mode='Testing'
              ## FOLD10 is reserved for testing
              logData=uscModel.test(batch_data,np.ones((uscData.mini_batch_size,1,1)))
         else:
              mode='Training'
              logData = uscModel.train(batch_data,np.ones((uscData.mini_batch_size,1,1)))
         uscLogger.appendLogData(stage_logs[mode],logData,False,mode) 
         uscLogger.appendLogData(epoch_logs[mode],logData,False,mode)
       uscLogger.logStepEnd(fold,mode,stage_logs,trainingIterationNo)

    uscLogger.logStepEnd('SUMMARY',mode,epoch_logs,trainingIterationNo)
    print(uscLogger.lastAccuracy)

if __name__ == '__main__':
 tf.compat.v1.app.run(main=main)

