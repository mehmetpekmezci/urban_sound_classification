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
     self.confusionMatrix=None
     self.lastAccuracy=0

       
                      
 def logStepStart(self,IterationNo):
    self.logger.info("###########################################################################################")
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       ('Epoch','Stage','Mode','Mean/Max/Min','Data Prep. Time','Duration','Loss','Accuracy')
                      )
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s" % 
                       ('---------------','---------------','---------------','---------------','---------------','---------------', '---------------','---------------')
                      )

 def printConfusionMatrix(self):
    self.logger.info("-------------------------------------")
    self.logger.info("Confusion Matrix for the first Classifier : ")
    self.logger.info("-------------------------------------")
    #self.logger.info("0 = air_conditioner, 1 = car_horn , 2 = children_playing , 3 = dog_bark , 4 = drilling , 5 = engine_idling , 6 = gun_shot , 7 = jackhammer , 8 = siren , 9 = street_music")
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s  " % 
                       ('air_conditioner',
                        'car_horn',
                        'children_playing',
                        'dog_bark',
                        'drilling',
                        'engine_idling',
                        'gun_shot',
                        'jackhammer',
                        'siren',
                        'street_music'
                        )
                      )

    for i in range(self.confusionMatrix.shape[0]) :
       self.logger.info("%-15d %-15d %-15d %-15d %-15d %-15d %-15d %-15d %-15d %-15d  " % 
                       (self.confusionMatrix[i][0], 
                        self.confusionMatrix[i][1],
                        self.confusionMatrix[i][2],
                        self.confusionMatrix[i][3],
                        self.confusionMatrix[i][4],
                        self.confusionMatrix[i][5],
                        self.confusionMatrix[i][6],
                        self.confusionMatrix[i][7],
                        self.confusionMatrix[i][8],
                        self.confusionMatrix[i][9]
                        )
                      )
          

    self.logger.info("")
    self.logger.info("")
   

 def printLog(self,stage,mode,logs,IterationNo):
    self.logger.info("%-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s %-15.15s"  % 
                       (IterationNo,stage,mode,'Mean',np.sum(logs[mode]['PrepareDataTimes']),np.sum(logs[mode]['Times']),
                        np.mean(logs[mode]['Losses']),np.mean(logs[mode]['Accuracies'])
                        )
                      )
                      
 def logStepEnd(self,stage,mode,logs,IterationNo):
 
    if stage == "SUMMARY" :
      mode='Training'
      self.printLog(stage,mode,logs,IterationNo)
      mode='Testing'
      self.printLog(stage,mode,logs,IterationNo)
      if self.confusionMatrix is not None :
          self.printConfusionMatrix()
          self.confusionMatrix=None
    else :
      self.printLog(stage,mode,logs,IterationNo)
    


 def getNewLogDictionaryCochlea(self):
    
    logs=dict()
    logs['Training']=dict()
    logs['Training']['PrepareDataTimes']=[]
    logs['Training']['Times']=[]
    logs['Training']['Accuracies']=[]
    logs['Training']['Losses']=[]
    logs['Testing']=dict()
    logs['Testing']['PrepareDataTimes']=[]
    logs['Testing']['Times']=[]
    logs['Testing']['Accuracies']=[]
    logs['Testing']['Losses']=[]
    
    return logs

 
  def appendLogDataCochlea(self,logDictionary,logData,mode):
         logDictionary['PrepareDataTimes'].append(logData[3])
         logDictionary['Times'].append(logData[0])
         logDictionary['Accuracies'].append(logData[2])
         logDictionary['Losses'].append(logData[1])
         if len(logData) == 5 :
            if self.confusionMatrix is None :
               self.confusionMatrix=logData[4]
            else:
               self.confusionMatrix=self.confusionMatrix+logData[4]

           
