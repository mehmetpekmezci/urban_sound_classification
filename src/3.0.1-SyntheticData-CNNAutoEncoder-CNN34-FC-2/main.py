#!/usr/bin/env python3
from header import *
from data import *
from NeuralNetworkModel import *
from AutoEncoder import *

def main(_):
  global fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION
  # if not already done  : Download from internet , convert to  csv files.   
  prepareData()
  # load all data into the memory
  MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION=load_all_np_data_back_to_memory(fold_data_dictionary)
  normalize_all_data(fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION)

  with tf.Session() as session:
    
   autoEncoder=AutoEncoder(session,logger)
   neuralNetworkModel=NeuralNetworkModel(session,logger,autoEncoder)
   saver = tf.train.Saver()


   ##
   ## INITIALIZE SESSION
   ##
   checkpoint= tf.train.get_checkpoint_state(os.path.dirname(SAVE_DIR+'/usc_model'))
   if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(session,checkpoint.model_checkpoint_path)
   else : 
    session.run(tf.global_variables_initializer())
   ##
   ##
    
   for iterationNo in range(AUTOENCODER_TRAINING_ITERATIONS):
        
    logger.info("##############################################################")
    logger.info("AutoEncoder Training Iteration : "+str(iterationNo))
        
    prepareDataTimes=[ ]
    trainingTimes=[ ]
    trainingAccuracies=[ ]
    testTimes=[ ]
    trainingLosses=[ ]
    for fold in np.random.permutation(FOLD_DIRS):
      ## DO NOT USE FOLD10 AT ALL
      if fold != "fold10": 
       current_fold_data=get_fold_data(fold)
       generated_data=generate_normalized_synthetic_samples(fold)
       for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
         if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
           batch_generated_data=generated_data[int(current_batch_counter*MINI_BATCH_SIZE_FOR_GENERATED_DATA):int((current_batch_counter+1)*MINI_BATCH_SIZE_FOR_GENERATED_DATA),:]
         else:
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
           batch_generated_data=generated_data[current_batch_counter*MINI_BATCH_SIZE_FOR_GENERATED_DATA:,:]
         trainingTime,trainingLoss,prepareDataTime=autoEncoder.train(batch_data,batch_generated_data)
         trainingTimes.append(trainingTime)
         trainingLosses.append(trainingLoss)
         prepareDataTimes.append(prepareDataTime)
    ## LOGGING            
    logger.info("AutoEncoder Prepare Data Time : "+str(np.sum(prepareDataTimes)))
    logger.info("AutoEncoder Training Time : "+str(np.sum(trainingTimes)))
    logger.info("AutoEncoder Mean Training Loss : "+str(np.mean(trainingLosses)))
    logger.info("AutoEncoder Max Training Loss : "+str(np.max(trainingLosses)))
    logger.info("AutoEncoder Min Training Loss : "+str(np.min(trainingLosses)))
    
    if iterationNo>0 and iterationNo%10 == 0 :
      saver.save(session, SAVE_DIR+'/usc_model-autoencoder',global_step=iterationNo)

       
   for trainingIterationNo in range(TRAINING_ITERATIONS):
        
    logger.info("##############################################################")
    logger.info("Training Iteration : "+str(trainingIterationNo))
        
    prepareDataTimes=[ ]
    trainingTimes=[ ]
    trainingAccuracies=[ ]
    testTimes=[ ]
    testAccuracies=[ ]
    UNIFIED_MINI_BATCH_SIZE=int(MINI_BATCH_SIZE+MINI_BATCH_SIZE_FOR_GENERATED_DATA)
    for fold in np.random.permutation(FOLD_DIRS):
       current_fold_data=get_fold_data(fold)
       for current_batch_counter in range(int(current_fold_data.shape[0]/UNIFIED_MINI_BATCH_SIZE)) :
         if (current_batch_counter+1)*UNIFIED_MINI_BATCH_SIZE <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*UNIFIED_MINI_BATCH_SIZE:(current_batch_counter+1)*UNIFIED_MINI_BATCH_SIZE,:]
         else:
           batch_data=current_fold_data[current_batch_counter*UNIFIED_MINI_BATCH_SIZE:,:]
         if fold == "fold10":
             ## FOLD10 is reserved for testing
              testTime,testAccuracy=neuralNetworkModel.test(batch_data)
              testTimes.append(testTime)
              testAccuracies.append(testAccuracy)
         else:
              trainingTime,trainingAccuracy,prepareDataTime=neuralNetworkModel.train(batch_data)
              trainingTimes.append(trainingTime)
              trainingAccuracies.append(trainingAccuracy)
              prepareDataTimes.append(prepareDataTime)



    ## LOGGING            
    logger.info("Prepare Data Time : "+str(np.sum(prepareDataTimes)))
    logger.info("Training Time : "+str(np.sum(trainingTimes)))
    logger.info("Mean Training Accuracy : "+str(np.mean(trainingAccuracies)))
    logger.info("Max Training Accuracy : "+str(np.max(trainingAccuracies)))
    logger.info("Min Training Accuracy : "+str(np.min(trainingAccuracies)))
    logger.info("Test Time : "+str(np.sum(testTimes)))
    logger.info("Mean Test Accuracy : "+str(np.mean(testAccuracies)))
    logger.info("Max Test Accuracy : "+str(np.max(testAccuracies)))
    logger.info("Min Test Accuracy : "+str(np.min(testAccuracies)))
    
    ## GRAPH (FOR LOGGING)
    tariningAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var: np.mean(trainingAccuracies)})
    trainingAccuracyWriter.add_summary(tariningAcuracySummary, trainingIterationNo)
    trainingAccuracyWriter.flush()

    testAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var:np.mean(testAccuracies)})
    testAccuracyWriter.add_summary(testAcuracySummary, trainingIterationNo)
    testAccuracyWriter.flush()

    tariningTimeSummary = session.run(tfSummaryTimeMergedWriter, {tf_summary_time_log_var: np.sum(trainingTimes)})
    trainingTimeWriter.add_summary(tariningTimeSummary, trainingIterationNo)
    trainingTimeWriter.flush()

    testTimeSummary = session.run(tfSummaryTimeMergedWriter, {tf_summary_time_log_var:np.mean(testTimes)})
    testTimeWriter.add_summary(testTimeSummary, trainingIterationNo)
    testTimeWriter.flush()

    if trainingIterationNo>0 and trainingIterationNo%10 == 0 :
      saver.save(session, SAVE_DIR+'/usc_model',global_step=trainingIterationNo)


if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,
                     default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





  
