#!/usr/bin/env python3
from header import *
from data import *

class TFSummary :
 def __init__(self,SCRIPT_DIR):
  self.SCRIPT_NAME = os.path.basename(SCRIPT_DIR)
  self.LOG_DIR_FOR_TF_SUMMARY=SCRIPT_DIR+"/../../logs/tf-summary/"+SCRIPT_NAME+"/"+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))
  if not os.path.exists(self.LOG_DIR_FOR_TF_SUMMARY):
    os.makedirs(self.LOG_DIR_FOR_TF_SUMMARY)

  ## ONE VARIABLE , TWO WRITERS TO OBTAIN TWO GRPAHS ON THE SAME IMAGE
  trainingAccuracyWriter = tf.summary.FileWriter(self.LOG_DIR_FOR_TF_SUMMARY+"/trainingAccuracyWriter")
  testAccuracyWriter =tf.summary.FileWriter(self.LOG_DIR_FOR_TF_SUMMARY+"/testAccuracyWriter")
  tf_summary_accuracy_log_var = tf.Variable(0.0)
  tf.summary.scalar("Accuracy (Test/Train)", tf_summary_accuracy_log_var)
  tfSummaryAccuracyMergedWriter = tf.summary.merge_all()

  trainingTimeWriter = tf.summary.FileWriter(self.LOG_DIR_FOR_TF_SUMMARY+"/trainingTimeWriter")
  testTimeWriter =tf.summary.FileWriter(self.LOG_DIR_FOR_TF_SUMMARY+"/testTimeWriter")
  tf_summary_time_log_var = tf.Variable(0.0)
  tf.summary.scalar("Time (Test/Train)", tf_summary_time_log_var)
  tfSummaryTimeMergedWriter = tf.summary.merge_all()



  
