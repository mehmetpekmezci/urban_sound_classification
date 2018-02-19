#!/usr/bin/env python3
from header import *
from data import *
from NeuralNetworkModel import *

def main(_):
    
  # if not already done  : Download from internet , convert to  csv files.   
  prepareData()
  # load all data into the memory
  load_all_csv_data_back_to_memory()
  # normalize all the data
  normalize_all_data()
  
  with tf.Session() as sess:
    
   neuralNetworkModel=NeuralNetworkModel(sess,logger)
    
   for trainingIterationNo in range(TRAINING_ITERATIONS):
        
    logger.info("Training Iteration : "+str(trainingIterationNo))
        
    trainingTimes=[ ]
    trainingAccuracies=[ ]
    testTimes=[ ]
    testAccuracies=[ ]
        
    for fold in np.random.permutation(fold_dirs):
       current_fold_data=get_fold_data(fold)
       for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)) :
         # MP asagidaki for dongusunde +1 olunca hatali tensor uretiyor tensorflow exception atiyor.
         #for current_batch_counter in range(int(current_fold_data.shape[0]/MINI_BATCH_SIZE)+1) :
         if (current_batch_counter+1)*MINI_BATCH_SIZE <= current_fold_data.shape[0] :
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:(current_batch_counter+1)*MINI_BATCH_SIZE,:]
         else:
           batch_data=current_fold_data[current_batch_counter*MINI_BATCH_SIZE:,:]
         if fold == "fold10":
             ## FOLD10 is reserved for testing
              testTime,testAccuracy=neuralNetworkModel.test(batch_data)
              testTimes.append(testTime)
              testAccuracies.append(testAccuracy)
         else:
              trainingTime,trainingAccuracy=neuralNetworkModel.train(batch_data)
              trainingTimes.append(trainingTime)
              trainingAccuracies.append(trainingAccuracy)



    ## LOGGING            
    logger.info("Training Time : "+str(np.sum(trainingTimes)))
    logger.info("Training Accuracy : "+str(np.mean(trainingAccuracies)))
    logger.info("Test Time : "+str(np.sum(testTimes)))
    logger.info("Test Accuracy : "+str(np.mean(testAccuracies)))
    
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

    testTimeSummary = session.run(tfSummaryTimeMergedWriter, {tf_summary_time_log_var:np.mean(testTimess)})
    testTimeWriter.add_summary(testTimeSummary, trainingIterationNo)
    testTimeWriter.flush()


if __name__ == '__main__':
 parser = argparse.ArgumentParser()
 parser.add_argument('--data_dir', type=str,
                     default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
 FLAGS, unparsed = parser.parse_known_args()
 tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





  
