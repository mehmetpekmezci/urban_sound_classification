#!/usr/bin/env python3
from header import *
from data import *
from NeuralNetworkModel import *

def main(_):
  global fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION
  # if not already done  : Download from internet , convert to  csv files.   
  prepareData()
  # load all data into the memory
  MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION=load_all_np_data_back_to_memory(fold_data_dictionary)
  # normalize all the data
#  logger.info("##############################################################")
#  logger.info("MIN_VALUE_FOR_NORMALIZATION : "+str(MIN_VALUE_FOR_NORMALIZATION))
#  logger.info("MAX_VALUE_FOR_NORMALIZATION : "+str(MAX_VALUE_FOR_NORMALIZATION))
#  logger.info("fold_data_dictionary['fold1'][0]: "+str(fold_data_dictionary['fold1'][0]))
#  logger.info("fold_data_dictionary['fold2'][0]: "+str(fold_data_dictionary['fold2'][0]))
#  logger.info("fold_data_dictionary['fold3'][0]: "+str(fold_data_dictionary['fold3'][0]))
#  logger.info("fold_data_dictionary['fold4'][0]: "+str(fold_data_dictionary['fold4'][0]))
#  logger.info("fold_data_dictionary['fold5'][0]: "+str(fold_data_dictionary['fold5'][0]))
#  logger.info("fold_data_dictionary['fold6'][0]: "+str(fold_data_dictionary['fold6'][0]))
#  logger.info("fold_data_dictionary['fold7'][0]: "+str(fold_data_dictionary['fold7'][0]))
#  logger.info("fold_data_dictionary['fold8'][0]: "+str(fold_data_dictionary['fold8'][0]))
#  logger.info("fold_data_dictionary['fold9'][0]: "+str(fold_data_dictionary['fold9'][0]))
#  logger.info("fold_data_dictionary['fold10'][0]: "+str(fold_data_dictionary['fold10'][0]))
#  logger.info("##############################################################")
  normalize_all_data(fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION)
  
#  logger.info("##############################################################")
#  logger.info("MIN_VALUE_FOR_NORMALIZATION : "+str(MIN_VALUE_FOR_NORMALIZATION))
#  logger.info("MAX_VALUE_FOR_NORMALIZATION : "+str(MAX_VALUE_FOR_NORMALIZATION))
#  logger.info("fold_data_dictionary['fold1'][0]: "+str(fold_data_dictionary['fold1'][0]))
#  logger.info("fold_data_dictionary['fold2'][0]: "+str(fold_data_dictionary['fold2'][0]))
#  logger.info("fold_data_dictionary['fold3'][0]: "+str(fold_data_dictionary['fold3'][0]))
#  logger.info("fold_data_dictionary['fold4'][0]: "+str(fold_data_dictionary['fold4'][0]))
#  logger.info("fold_data_dictionary['fold5'][0]: "+str(fold_data_dictionary['fold5'][0]))
#  logger.info("fold_data_dictionary['fold6'][0]: "+str(fold_data_dictionary['fold6'][0]))
#  logger.info("fold_data_dictionary['fold7'][0]: "+str(fold_data_dictionary['fold7'][0]))
#  logger.info("fold_data_dictionary['fold8'][0]: "+str(fold_data_dictionary['fold8'][0]))
#  logger.info("fold_data_dictionary['fold9'][0]: "+str(fold_data_dictionary['fold9'][0]))
#  logger.info("fold_data_dictionary['fold10'][0]: "+str(fold_data_dictionary['fold10'][0]))
#  logger.info("##############################################################")
  


  with tf.Session() as session:
    
   neuralNetworkModel=NeuralNetworkModel(session,logger)
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
    
       
   for trainingIterationNo in range(TRAINING_ITERATIONS):
        
    logger.info("##############################################################")
    logger.info("Training Iteration : "+str(trainingIterationNo))
        
    prepareDataTimes=[ ]
    trainingTimes=[ ]
    trainingAccuracies_1=[ ]
    trainingAccuracies_2=[ ]
    trainingAccuracies_metric=[ ]
    testTimes=[ ]
    testAccuracies_1=[ ]
    testAccuracies_2=[ ]
    testAccuracies_metric=[ ]
    tainingLosses_metric=[]
    for fold in np.random.permutation(FOLD_DIRS):
       if fold == "fold10":
         logger.info(" Starting Fold Testing : "+fold)
       else :
         logger.info(" Starting Fold Training : "+fold)
       current_fold_data=get_fold_data(fold)
       for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
         #if current_batch_counter % 10 == 0 :
         #  logger.info(" Batch Counter : "+str(current_batch_counter))
         if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
         else:
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
         if fold == "fold10":
             ## FOLD10 is reserved for testing
              testTime,testAccuracy_1,testAccuracy_2,testAccuracy_metric=neuralNetworkModel.test(batch_data)
              testTimes.append(testTime)
              testAccuracies_1.append(testAccuracy_1)
              testAccuracies_2.append(testAccuracy_2)
              testAccuracies_metric.append(testAccuracy_metric)
         else:
              batch_data_1=batch_data
              batch_data_2=np.random.permutation(batch_data)
              #batch_data_2=np.copy(batch_data)
              #for i in range(int(batch_data_2.shape[0]/2)):
              #   batch_data_2[i]=batch_data_2[i+1]
              trainingTime,trainingAccuracy_1,trainingAccuracy_2,trainingAccuracy_metric,loss_adverserial,prepareDataTime=neuralNetworkModel.train(batch_data_1,batch_data_2)
              trainingTimes.append(trainingTime)
              trainingAccuracies_1.append(trainingAccuracy_1)
              trainingAccuracies_2.append(trainingAccuracy_2)
              trainingAccuracies_metric.append(trainingAccuracy_metric)
              tainingLosses_metric.append(loss_adverserial)
              
              prepareDataTimes.append(prepareDataTime)



    ## LOGGING            
    logger.info("Prepare Data Time : "+str(np.sum(prepareDataTimes)))
    logger.info("Training Time : "+str(np.sum(trainingTimes)))
    logger.info("Mean Training Accuracy_1 : "+str(np.mean(trainingAccuracies_1)))
    logger.info("Max Training Accuracy_1 : "+str(np.max(trainingAccuracies_1)))
    logger.info("Min Training Accuracy_1 : "+str(np.min(trainingAccuracies_1)))
    logger.info("Mean Training Accuracy_2 : "+str(np.mean(trainingAccuracies_2)))
    logger.info("Max Training Accuracy_2 : "+str(np.max(trainingAccuracies_2)))
    logger.info("Min Training Accuracy_2 : "+str(np.min(trainingAccuracies_2)))
    logger.info("Mean Training Accuracy_metric : "+str(np.mean(trainingAccuracies_metric)))
    logger.info("Max Training Accuracy_metric : "+str(np.max(trainingAccuracies_metric)))
    logger.info("Min Training Accuracy_metric : "+str(np.min(trainingAccuracies_metric)))
    logger.info("Mean Training Loss Metric : "+str(np.mean(tainingLosses_metric)))
    logger.info("Max Training  Loss Metric : "+str(np.max(tainingLosses_metric)))
    logger.info("Min Training  Loss Metric : "+str(np.min(tainingLosses_metric)))
    logger.info("Test Time : "+str(np.sum(testTimes)))
    if len(testAccuracies_1) > 0 :
      logger.info("Mean Test Accuracy_1 : "+str(np.mean(testAccuracies_1)))
      logger.info("Max Test Accuracy_1 : "+str(np.max(testAccuracies_1)))
      logger.info("Min Test Accuracy_1 : "+str(np.min(testAccuracies_1)))
      logger.info("Mean Test Accuracy_2 : "+str(np.mean(testAccuracies_2)))
      logger.info("Max Test Accuracy_2 : "+str(np.max(testAccuracies_2)))
      logger.info("Min Test Accuracy_2 : "+str(np.min(testAccuracies_2)))
      logger.info("Mean Test Accuracy_metric : "+str(np.mean(testAccuracies_metric)))
      logger.info("Max Test Accuracy_metric : "+str(np.max(testAccuracies_metric)))
      logger.info("Min Test Accuracy_metric : "+str(np.min(testAccuracies_metric)))
    
    ## GRAPH (FOR LOGGING)
    tariningAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var: np.mean(trainingAccuracies_1)})
    trainingAccuracyWriter.add_summary(tariningAcuracySummary, trainingIterationNo)
    trainingAccuracyWriter.flush()

    tariningAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var: np.mean(trainingAccuracies_2)})
    trainingAccuracyWriter.add_summary(tariningAcuracySummary, trainingIterationNo)
    trainingAccuracyWriter.flush()

    testAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var:np.mean(testAccuracies_1)})
    testAccuracyWriter.add_summary(testAcuracySummary, trainingIterationNo)
    testAccuracyWriter.flush()

    testAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var:np.mean(testAccuracies_2)})
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





  
