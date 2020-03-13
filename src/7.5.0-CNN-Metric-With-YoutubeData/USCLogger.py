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
     #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
     loggingFileHandler.setFormatter(formatter)
     loggingConsoleHandler.setFormatter(formatter)
     self.logger.addHandler(loggingFileHandler)
     self.logger.addHandler(loggingConsoleHandler)


     ##
     ## CONFIGURE TF.SUMMARY
     ##
     ## ONE VARIABLE , TWO WRITERS TO OBTAIN TWO GRPAHS ON THE SAME IMAGE
     #self.trainingAccuracyWriter = tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/trainingAccuracyWriter")
     #self.testAccuracyWriter =tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/testAccuracyWriter")
     #self.tf_summary_accuracy_log_var = tf.Variable(0.0)
     #tf.compat.v1.summary.scalar("Accuracy-Test-Train", self.tf_summary_accuracy_log_var)
     #self.tfSummaryAccuracyMergedWriter = tf.compat.v1.summary.merge_all()

     #self.trainingTimeWriter = tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/trainingTimeWriter")
     #self.testTimeWriter =tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/testTimeWriter")
     #self.tf_summary_time_log_var = tf.Variable(0.0)
     #tf.compat.v1.summary.scalar("Time-Test-Train", self.tf_summary_time_log_var)
     #self.tfSummaryTimeMergedWriter = tf.compat.v1.summary.merge_all()

 def logStepStart(self,trainingIterationNo):
    self.logger.info("###########################################################################################")
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " % 
                       ('Epoch','Stage','Test/Train','Mean/Max/Min','Data Prep. Time','Duration',
                        'Total Loss', 
                        'Class. 1 Loss','Class. 1 Acc.',
                        'Class. 2 Loss','Class. 2 Acc.',
                        'AutoEnc. 1 Loss','AutoEnc. 1 Acc.',
                        'AutoEnc. 2 Loss','AutoEnc. 2 Acc.',
                        'Discrim. Loss','Discrim. Acc.'
                        )
                      )
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " % 
                       ('---------------','---------------','---------------','---------------','---------------','---------------',
                        '---------------', 
                        '---------------','---------------',
                        '---------------','---------------',
                        '---------------','---------------',
                        '---------------','---------------',
                        '---------------','---------------'
                        )
                      )



 def logStepEnd(self,stage,logs,trainingIterationNo):
 
   if len(logs['testTimes'])>0 :

    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " % 
                       (trainingIterationNo,stage,'Training','Mean',np.sum(logs['trainingPrepareDataTimes']),np.sum(logs['trainingTimes']),
                        np.mean(logs['trainingLosses_total']), 
                        np.mean(logs['trainingLosses_classifier_1']),np.mean(logs['trainingAccuracies_classifier_1']),
                        np.mean(logs['trainingLosses_classifier_2']),np.mean(logs['trainingAccuracies_classifier_2']),
                        np.mean(logs['trainingLosses_autoencoder_1']),np.mean(logs['trainingAccuracies_autoencoder_1']),
                        np.mean(logs['trainingLosses_autoencoder_2']),np.mean(logs['trainingAccuracies_autoencoder_2']),
                        np.mean(logs['trainingLosses_discriminator']),np.mean(logs['trainingAccuracies_discriminator'])
                        )
                      )
 
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       (trainingIterationNo,stage,'Training','Max',np.sum(logs['trainingPrepareDataTimes']),np.sum(logs['trainingTimes']),
                        np.max(logs['trainingLosses_total']), 
                        np.max(logs['trainingLosses_classifier_1']),np.max(logs['trainingAccuracies_classifier_1']),
                        np.max(logs['trainingLosses_classifier_2']),np.max(logs['trainingAccuracies_classifier_2']),
                        np.max(logs['trainingLosses_autoencoder_1']),np.max(logs['trainingAccuracies_autoencoder_1']),
                        np.max(logs['trainingLosses_autoencoder_2']),np.max(logs['trainingAccuracies_autoencoder_2']),
                        np.max(logs['trainingLosses_discriminator']),np.max(logs['trainingAccuracies_discriminator'])
                        )
                      )
 
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       (trainingIterationNo,stage,'Training','Min',np.sum(logs['trainingPrepareDataTimes']),np.sum(logs['trainingTimes']),
                        np.min(logs['trainingLosses_total']), 
                        np.min(logs['trainingLosses_classifier_1']),np.min(logs['trainingAccuracies_classifier_1']),
                        np.min(logs['trainingLosses_classifier_2']),np.min(logs['trainingAccuracies_classifier_2']),
                        np.min(logs['trainingLosses_autoencoder_1']),np.min(logs['trainingAccuracies_autoencoder_1']),
                        np.min(logs['trainingLosses_autoencoder_2']),np.min(logs['trainingAccuracies_autoencoder_2']),
                        np.min(logs['trainingLosses_discriminator']),np.min(logs['trainingAccuracies_discriminator'])
                        )
                      )

   if len(logs['testTimes'])>0 :
    
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       (trainingIterationNo,stage,'Test','Mean',np.sum(logs['testPrepareDataTimes']),np.sum(logs['testTimes']),
                        np.mean(logs['testLosses_total']), 
                        np.mean(logs['testLosses_classifier_1']),np.mean(logs['testAccuracies_classifier_1']),
                        np.mean(logs['testLosses_classifier_2']),np.mean(logs['testAccuracies_classifier_2']),
                        np.mean(logs['testLosses_autoencoder_1']),np.mean(logs['testAccuracies_autoencoder_1']),
                        np.mean(logs['testLosses_autoencoder_2']),np.mean(logs['testAccuracies_autoencoder_2']),
                        np.mean(logs['testLosses_discriminator']),np.mean(logs['testAccuracies_discriminator'])
                        )
                      )
 
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       (trainingIterationNo,stage,'Test','Max',np.sum(logs['testPrepareDataTimes']),np.sum(logs['testTimes']),
                        np.max(logs['testLosses_total']), 
                        np.max(logs['testLosses_classifier_1']),np.max(logs['testAccuracies_classifier_1']),
                        np.max(logs['testLosses_classifier_2']),np.max(logs['testAccuracies_classifier_2']),
                        np.max(logs['testLosses_autoencoder_1']),np.max(logs['testAccuracies_autoencoder_1']),
                        np.max(logs['testLosses_autoencoder_2']),np.max(logs['testAccuracies_autoencoder_2']),
                        np.max(logs['testLosses_discriminator']),np.max(logs['testAccuracies_discriminator'])
                        )
                      )
 
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       (trainingIterationNo,stage,'Test','Min',np.sum(logs['testPrepareDataTimes']),np.sum(logs['testTimes']),
                        np.min(logs['testLosses_total']), 
                        np.min(logs['testLosses_classifier_1']),np.min(logs['testAccuracies_classifier_1']),
                        np.min(logs['testLosses_classifier_2']),np.min(logs['testAccuracies_classifier_2']),
                        np.min(logs['testLosses_autoencoder_1']),np.min(logs['testAccuracies_autoencoder_1']),
                        np.min(logs['testLosses_autoencoder_2']),np.min(logs['testAccuracies_autoencoder_2']),
                        np.min(logs['testLosses_discriminator']),np.min(logs['testAccuracies_discriminator'])
                        )
                      )



    ## GRAPH (FOR LOGGING)
    #tariningAcuracySummary = session.run(self.tfSummaryAccuracyMergedWriter, {self.tf_summary_accuracy_log_var: np.mean(trainingAccuracies)})
    #self.trainingAccuracyWriter.add_summary(tariningAcuracySummary, trainingIterationNo)
    #self.trainingAccuracyWriter.flush()

    #testAcuracySummary = session.run(self.tfSummaryAccuracyMergedWriter, {self.tf_summary_accuracy_log_var:np.mean(testAccuracies)})
    #self.testAccuracyWriter.add_summary(testAcuracySummary, trainingIterationNo)
    #self.testAccuracyWriter.flush()

    #tariningTimeSummary = session.run(self.tfSummaryTimeMergedWriter, {self.tf_summary_time_log_var: np.sum(trainingTimes)})
    #self.trainingTimeWriter.add_summary(tariningTimeSummary, trainingIterationNo)
    #self.trainingTimeWriter.flush()

    #testTimeSummary = session.run(self.tfSummaryTimeMergedWriter, {self.tf_summary_time_log_var:np.mean(testTimes)})
    #self.testTimeWriter.add_summary(testTimeSummary, trainingIterationNo)
    #self.testTimeWriter.flush()






