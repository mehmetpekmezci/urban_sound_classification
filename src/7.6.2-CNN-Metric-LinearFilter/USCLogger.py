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
     #self.AccuracyWriter = tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/AccuracyWriter")
     #self.AccuracyWriter =tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/AccuracyWriter")
     #self.tf_summary_accuracy_log_var = tf.Variable(0.0)
     #tf.compat.v1.summary.scalar("Accuracy-Test-Train", self.tf_summary_accuracy_log_var)
     #self.tfSummaryAccuracyMergedWriter = tf.compat.v1.summary.merge_all()

     #self.TimeWriter = tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/TimeWriter")
     #self.TimeWriter =tf.compat.v1.summary.FileWriter(self.log_dir_for_tfsummary+"/TimeWriter")
     #self.tf_summary_time_log_var = tf.Variable(0.0)
     #tf.compat.v1.summary.scalar("Time-Test-Train", self.tf_summary_time_log_var)
     #self.tfSummaryTimeMergedWriter = tf.compat.v1.summary.merge_all()

 def logStepStart(self,IterationNo):
    self.logger.info("###########################################################################################")
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " % 
                       ('Epoch','Stage','Mode','Mean/Max/Min','Data Prep. Time','Duration',
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


 def printLog(self,stage,mode,logs,IterationNo):
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f %-15.5f " % 
                       (IterationNo,stage,mode,'Mean',np.sum(logs[mode]['PrepareDataTimes']),np.sum(logs[mode]['Times']),
                        np.mean(logs[mode]['Losses_total']), 
                        np.mean(logs[mode]['Losses_classifier_1']),np.mean(logs[mode]['Accuracies_classifier_1']),
                        np.mean(logs[mode]['Losses_classifier_2']),np.mean(logs[mode]['Accuracies_classifier_2']),
                        np.mean(logs[mode]['Losses_autoencoder_1']),np.mean(logs[mode]['Accuracies_autoencoder_1']),
                        np.mean(logs[mode]['Losses_autoencoder_2']),np.mean(logs[mode]['Accuracies_autoencoder_2']),
                        np.mean(logs[mode]['Losses_discriminator']),np.mean(logs[mode]['Accuracies_discriminator'])
                        )
                      )
                      
 def logStepEnd(self,stage,mode,logs,IterationNo):
 
    if stage == "SUMMARY" :
      mode='Training'
      self.printLog(stage,mode,logs,IterationNo)
      mode='Testing'
      self.printLog(stage,mode,logs,IterationNo)
    else :
      self.printLog(stage,mode,logs,IterationNo)
    


    ## GRAPH (FOR LOGGING)
    #tariningAcuracySummary = session.run(self.tfSummaryAccuracyMergedWriter, {self.tf_summary_accuracy_log_var: np.mean(Accuracies)})
    #self.AccuracyWriter.add_summary(tariningAcuracySummary, IterationNo)
    #self.AccuracyWriter.flush()

    #AcuracySummary = session.run(self.tfSummaryAccuracyMergedWriter, {self.tf_summary_accuracy_log_var:np.mean(Accuracies)})
    #self.AccuracyWriter.add_summary(AcuracySummary, IterationNo)
    #self.AccuracyWriter.flush()

    #tariningTimeSummary = session.run(self.tfSummaryTimeMergedWriter, {self.tf_summary_time_log_var: np.sum(Times)})
    #self.TimeWriter.add_summary(tariningTimeSummary, IterationNo)
    #self.TimeWriter.flush()

    #TimeSummary = session.run(self.tfSummaryTimeMergedWriter, {self.tf_summary_time_log_var:np.mean(Times)})
    #self.TimeWriter.add_summary(TimeSummary, IterationNo)
    #self.TimeWriter.flush()


 def getNewLogDictionary(self):
    logs=dict()
    logs['Training']=dict()
    logs['Training']['PrepareDataTimes']=[]
    logs['Training']['Times']=[]
    logs['Training']['Accuracies_classifier_1']=[]
    logs['Training']['Accuracies_classifier_2']=[]
    logs['Training']['Accuracies_autoencoder_1']=[]
    logs['Training']['Accuracies_autoencoder_2']=[]
    logs['Training']['Accuracies_discriminator']=[]
    logs['Training']['Losses_total']=[]
    logs['Training']['Losses_classifier_1']=[]
    logs['Training']['Losses_classifier_2']=[]
    logs['Training']['Losses_autoencoder_1']=[]
    logs['Training']['Losses_autoencoder_2']=[]
    logs['Training']['Losses_discriminator']=[]
    logs['Testing']=dict()
    logs['Testing']['PrepareDataTimes']=[]
    logs['Testing']['Times']=[]
    logs['Testing']['Accuracies_classifier_1']=[]
    logs['Testing']['Accuracies_classifier_2']=[]
    logs['Testing']['Accuracies_autoencoder_1']=[]
    logs['Testing']['Accuracies_autoencoder_2']=[]
    logs['Testing']['Accuracies_discriminator']=[]
    logs['Testing']['Losses_total']=[]
    logs['Testing']['Losses_classifier_1']=[]
    logs['Testing']['Losses_classifier_2']=[]
    logs['Testing']['Losses_autoencoder_1']=[]
    logs['Testing']['Losses_autoencoder_2']=[]
    logs['Testing']['Losses_discriminator']=[]
    

    
    return logs

 def appendLogData(self,logDictionary,logData):
 
         logDictionary['PrepareDataTimes'].append(logData[12])
         logDictionary['Times'].append(logData[0])
         #if logData[7] >0 and logData[7] <= 1 :
         logDictionary['Accuracies_classifier_1'].append(logData[7])
         #if logData[8] >0 and logData[8] <= 1 :
         logDictionary['Accuracies_classifier_2'].append(logData[8])
         logDictionary['Accuracies_autoencoder_1'].append(logData[9])
         logDictionary['Accuracies_autoencoder_2'].append(logData[10])
         logDictionary['Accuracies_discriminator'].append(logData[11])
         logDictionary['Losses_total'].append(logData[1])
         #if logData[2] >0  :
         logDictionary['Losses_classifier_1'].append(logData[2])
         #if logData[2] >0  :
         logDictionary['Losses_classifier_2'].append(logData[3])
         logDictionary['Losses_autoencoder_1'].append(logData[4])
         logDictionary['Losses_autoencoder_2'].append(logData[5])
         logDictionary['Losses_discriminator'].append(logData[6])



