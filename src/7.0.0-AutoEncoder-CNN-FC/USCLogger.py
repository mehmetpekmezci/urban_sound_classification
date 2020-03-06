#!/usr/bin/env python3
from USCHeader import *

class USCLogger :
 def __init__(self):
     self.script_dir=os.path.dirname(os.path.realpath(__file__))
     self.script_name=os.path.basename(self.script_dir)
     self.log_dir_for_logger=self.script_dir+"/../../logs/logger/"+self.script_name
     self.log_dir_for_tfsummary=self.script_dir+"/../../logs/tf-summary/"+self.script_name+"/"+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))

     if not os.path.exists(self.log_dir_for_tfsummary):
         os.makedirs(self.log_dir_for_tfsummary)
     if not os.path.exists(self.log_dir_for_logger):
         os.makedirs(self.log_dir_for_logger)
    
     ## CONFUGRE LOGGING
     self.logger=logging.getLogger('usc')
     self.logger.propagate=False
     self.logger.setLevel(logging.INFO)
     loggingFileHandler = logging.FileHandler(self.log_dir_for_logger+'/usc-'+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))+'.log')
     loggingFileHandler.setLevel(logging.DEBUG)
     loggingConsoleHandler = logging.StreamHandler()
     loggingConsoleHandler.setLevel(logging.DEBUG)
     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     loggingFileHandler.setFormatter(formatter)
     loggingConsoleHandler.setFormatter(formatter)
     self.logger.addHandler(loggingFileHandler)
     self.logger.addHandler(loggingConsoleHandler)


     ##
     ## CONFIGURE TF.SUMMARY
     ##
     ## ONE VARIABLE , TWO WRITERS TO OBTAIN TWO GRPAHS ON THE SAME IMAGE
     self.trainingAccuracyWriter = tf.summary.FileWriter(self.log_dir_for_tfsummary+"/trainingAccuracyWriter")
     self.testAccuracyWriter =tf.summary.FileWriter(self.log_dir_for_tfsummary+"/testAccuracyWriter")
     self.tf_summary_accuracy_log_var = tf.Variable(0.0)
     tf.summary.scalar("Accuracy-Test-Train", self.tf_summary_accuracy_log_var)
     self.tfSummaryAccuracyMergedWriter = tf.summary.merge_all()

     self.trainingTimeWriter = tf.summary.FileWriter(self.log_dir_for_tfsummary+"/trainingTimeWriter")
     self.testTimeWriter =tf.summary.FileWriter(self.log_dir_for_tfsummary+"/testTimeWriter")
     self.tf_summary_time_log_var = tf.Variable(0.0)
     tf.summary.scalar("Time-Test-Train", self.tf_summary_time_log_var)
     self.tfSummaryTimeMergedWriter = tf.summary.merge_all()

 def logStepStart(self,session,trainingIterationNo):
    self.logger.info("##############################################################")
    self.logger.info("Training Iteration : "+str(trainingIterationNo))

 def logStepEnd(self,session,prepareDataTimes,trainingTimes,trainingAccuracies,testTimes,testAccuracies,trainingIterationNo):
    self.logger.info("Prepare Data Time : "+str(np.sum(prepareDataTimes)))
    self.logger.info("Training Time : "+str(np.sum(trainingTimes)))
    self.logger.info("Mean Training Accuracy : "+str(np.mean(trainingAccuracies)))
    self.logger.info("Max Training Accuracy : "+str(np.max(trainingAccuracies)))
    self.logger.info("Min Training Accuracy : "+str(np.min(trainingAccuracies)))
    if len(testAccuracies) > 0 :
     self.logger.info("Test Time : "+str(np.sum(testTimes)))
     self.logger.info("Mean Test Accuracy : "+str(np.mean(testAccuracies)))
     self.logger.info("Max Test Accuracy : "+str(np.max(testAccuracies)))
     self.logger.info("Min Test Accuracy : "+str(np.min(testAccuracies)))
    ## GRAPH (FOR LOGGING)
    tariningAcuracySummary = session.run(self.tfSummaryAccuracyMergedWriter, {self.tf_summary_accuracy_log_var: np.mean(trainingAccuracies)})
    self.trainingAccuracyWriter.add_summary(tariningAcuracySummary, trainingIterationNo)
    self.trainingAccuracyWriter.flush()

    testAcuracySummary = session.run(self.tfSummaryAccuracyMergedWriter, {self.tf_summary_accuracy_log_var:np.mean(testAccuracies)})
    self.testAccuracyWriter.add_summary(testAcuracySummary, trainingIterationNo)
    self.testAccuracyWriter.flush()

    tariningTimeSummary = session.run(self.tfSummaryTimeMergedWriter, {self.tf_summary_time_log_var: np.sum(trainingTimes)})
    self.trainingTimeWriter.add_summary(tariningTimeSummary, trainingIterationNo)
    self.trainingTimeWriter.flush()

    testTimeSummary = session.run(self.tfSummaryTimeMergedWriter, {self.tf_summary_time_log_var:np.mean(testTimes)})
    self.testTimeWriter.add_summary(testTimeSummary, trainingIterationNo)
    self.testTimeWriter.flush()


 def logAutoEncoderStepStart(self,session,trainingIterationNo):
    self.logger.info("##############################################################")
    self.logger.info("AutoEncoder Training Iteration : "+str(trainingIterationNo))

 def logAutoEncoderStepEnd(self,session,prepareDataTimes,trainingTimes,trainingLosses,accuracies,trainingIterationNo):
    self.logger.info("AutoEncoder Prepare Data Time : "+str(np.sum(prepareDataTimes)))
    self.logger.info("AutoEncoder Training Time : "+str(np.sum(trainingTimes)))
    self.logger.info("AutoEncoder Mean Accuracies : "+str(np.mean(accuracies)))
    self.logger.info("AutoEncoder Mean Training Loss : "+str(np.mean(trainingLosses)))
    self.logger.info("AutoEncoder Max Training Loss : "+str(np.max(trainingLosses)))
    self.logger.info("AutoEncoder Min Training Loss : "+str(np.min(trainingLosses)))





