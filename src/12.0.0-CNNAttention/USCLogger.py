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
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " % 
                       ('Epoch','Stage','Mode','Mean/Max/Min','Data Prep. Time','Duration', 'Loss', 'Acc.')
                      )
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " %
                       ('---------------','---------------','---------------','---------------','---------------','---------------', '---------------','---------------')
                      )


 def printLog(self,stage,mode,logs,IterationNo):
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s " %
                       (IterationNo,stage,mode,'Mean',np.sum(logs[mode]['PrepareDataTimes']),np.sum(logs[mode]['Times']), np.mean(logs[mode]['Loss']), np.mean(logs[mode]['Accuracy']))
                      )
                      
 def logStepEnd(self,stage,mode,logs,IterationNo):
 
    if stage == "SUMMARY" :
      mode='Training'
      self.printLog(stage,mode,logs,IterationNo)
      mode='Testing'
      self.printLog(stage,mode,logs,IterationNo)
    else :
      self.printLog(stage,mode,logs,IterationNo)
    

 def getNewLogDictionary(self):
    logs=dict()
    logs['Training']=dict()
    logs['Training']['PrepareDataTimes']=[]
    logs['Training']['Times']=[]
    logs['Training']['Accuracy']=[]
    logs['Training']['Loss']=[]
    logs['Testing']=dict()
    logs['Testing']['PrepareDataTimes']=[]
    logs['Testing']['Times']=[]
    logs['Testing']['Accuracy']=[]
    logs['Testing']['Loss']=[]
    
    return logs

 def appendLogData(self,logDictionary,logData):
         logDictionary['Times'].append(logData[0])
         logDictionary['Loss'].append(logData[1])
         logDictionary['Accuracy'].append(logData[2])
         logDictionary['PrepareDataTimes'].append(logData[3])



