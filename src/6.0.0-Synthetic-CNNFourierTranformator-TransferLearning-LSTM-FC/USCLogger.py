#!/usr/bin/env python3
from header import *
from data import *

class USCLogger :
 def __init__(self,SCRIPT_DIR):
  self.SCRIPT_NAME = os.path.basename(SCRIPT_DIR)
  self.LOG_DIR_FOR_LOGGER=SCRIPT_DIR+"/../../logs/logger/"+self.SCRIPT_NAME
  if not os.path.exists(self.LOG_DIR_FOR_LOGGER):
    os.makedirs(self.LOG_DIR_FOR_LOGGER)
        
  ## CONFUGRE LOGGING
  logger=logging.getLogger("usc")
  logger.setLevel(logging.INFO)
  loggingFileHandler = logging.FileHandler(self.LOG_DIR_FOR_LOGGER+'/usc-'+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))+'.log')
  loggingFileHandler.setLevel(logging.DEBUG)
  loggingConsoleHandler = logging.StreamHandler()
  loggingConsoleHandler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  loggingFileHandler.setFormatter(formatter)
  loggingConsoleHandler.setFormatter(formatter)
  logger.addHandler(loggingFileHandler)
  logger.addHandler(loggingConsoleHandler)
  self.logger=logger
  
