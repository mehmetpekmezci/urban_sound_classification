#!/usr/bin/env python3
from header import *
from data import *
from NeuralNetworkModel import *

def main(_):
  global fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION
  prepareData()
  MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION=load_all_np_data_back_to_memory(fold_data_dictionary)
  normalize_all_data(fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION)

  with tf.Session() as session:
   neuralNetworkModel=NeuralNetworkModel(session,logger)
   saver = tf.train.Saver()
   checkpoint= tf.train.get_checkpoint_state(os.path.dirname(SAVE_DIR+'/usc_model'))
   if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(session,checkpoint.model_checkpoint_path)
   else : 
    session.run(tf.global_variables_initializer())
       
   for trainingIterationNo in range(TRAINING_ITERATIONS):
    logger.info("##############################################################")
    logger.info("Training Iteration : "+str(trainingIterationNo))
    prepareDataTimes=[ ]
    trainingTimes=[ ]
    trainingAccuracies=[ ]
    testTimes=[ ]
    testAccuracies=[ ]
    tainingLosses=[]
    for fold in np.random.permutation(FOLD_DIRS):
       if fold == "fold10":
         logger.info(" Starting Fold Testing : "+fold)
       else :
         logger.info(" Starting Fold Training : "+fold)
       current_fold_data=get_fold_data(fold)
       for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
         if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
         else:
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
         if fold == "fold10":
             ## FOLD10 is reserved for testing
              batch_data=np.random.permutation(batch_data)
              testTime,testAccuracy=neuralNetworkModel.test(batch_data)
              testTimes.append(testTime)
              testAccuracies.append(testAccuracy)
         else:
              batch_data=np.random.permutation(batch_data)
              trainingTime,trainingAccuracy,loss,prepareDataTime=neuralNetworkModel.train(batch_data)
              trainingTimes.append(trainingTime)
              trainingAccuracies.append(trainingAccuracy)
              tainingLosses.append(loss)
              prepareDataTimes.append(prepareDataTime)
    ## LOGGING            
    logger.info("Prepare Data Time : "+str(np.sum(prepareDataTimes)))
    logger.info("Training Time : "+str(np.sum(trainingTimes)))
    logger.info("Mean Training Loss : "+str(np.mean(tainingLosses)))
    logger.info("Mean Training Accuracy : "+str(np.mean(trainingAccuracies)))
    if len(testAccuracies) > 0 :
      logger.info("Mean Test Accuracy : "+str(np.mean(testAccuracies)))
    
    ## GRAPH (FOR LOGGING)
    tariningAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var: np.mean(trainingAccuracies)})
    trainingAccuracyWriter.add_summary(tariningAcuracySummary, trainingIterationNo)
    trainingAccuracyWriter.flush()
    testAcuracySummary = session.run(tfSummaryAccuracyMergedWriter, {tf_summary_accuracy_log_var:np.mean(testAccuracies)})
    testAccuracyWriter.add_summary(testAcuracySummary, trainingIterationNo)
    testAccuracyWriter.flush()
    if trainingIterationNo>0 and trainingIterationNo%10 == 0 :
      saver.save(session, SAVE_DIR+'/usc_model',global_step=trainingIterationNo)

if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





