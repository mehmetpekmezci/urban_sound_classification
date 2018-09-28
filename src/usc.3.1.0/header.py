#!/usr/bin/env python3
##
## IMPORTS
##
import importlib
math        = importlib.import_module("math")
logging     = importlib.import_module("logging")
tf          = importlib.import_module("tensorflow")
urllib3     = importlib.import_module("urllib3")
tarfile     = importlib.import_module("tarfile")
csv         = importlib.import_module("csv")
glob        = importlib.import_module("glob")
sys         = importlib.import_module("sys")
os          = importlib.import_module("os")
argparse    = importlib.import_module("argparse")
np          = importlib.import_module("numpy")
librosa     = importlib.import_module("librosa")
pandas      = importlib.import_module("pandas")
time        = importlib.import_module("time")
random      = importlib.import_module("random")
datetime    = importlib.import_module("datetime")


##
## DEFAULT CONFIGS
##

# uncomment below for debugging
#np.set_printoptions(threshold=np.nan)

##
## DATA DIRECTORY NAMES
##
FOLD_DIRS = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
MAIN_DATA_DIR = SCRIPT_DIR+'/../../data/'
RAW_DATA_DIR = MAIN_DATA_DIR+'/0.raw/UrbanSound8K/audio'
CSV_DATA_DIR=MAIN_DATA_DIR+"/1.csv"
LOG_DIR_FOR_LOGGER=SCRIPT_DIR+"/../../logs/logger"
LOG_DIR_FOR_TF_SUMMARY=SCRIPT_DIR+"/../../logs/tf-summary"


if not os.path.exists(LOG_DIR_FOR_TF_SUMMARY):
    os.makedirs(LOG_DIR_FOR_TF_SUMMARY)
if not os.path.exists(LOG_DIR_FOR_LOGGER):
    os.makedirs(LOG_DIR_FOR_LOGGER)
    
    
## CONFUGRE LOGGING
logger=logging.getLogger('usc')
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
# create file handler and set level to debug
loggingFileHandler = logging.FileHandler(LOG_DIR_FOR_LOGGER+'/usc-'+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))+'.log')
loggingFileHandler.setLevel(logging.DEBUG)
# create console handler and set level to debug
loggingConsoleHandler = logging.StreamHandler()
loggingConsoleHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
loggingFileHandler.setFormatter(formatter)
loggingConsoleHandler.setFormatter(formatter)
logger.addHandler(loggingFileHandler)
logger.addHandler(loggingConsoleHandler)
#logger.debug('debug message')
#logger.info('info message')
#logger.warn('warn message')
#logger.error('error message')
#logger.critical('critical message')

##
## CONFIGURE TF.SUMMARY
##
## ONE VARIABLE , TWO WRITERS TO OBTAIN TWO GRPAHS ON THE SAME IMAGE
trainingAccuracyWriter = tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY+"/trainingAccuracyWriter")
testAccuracyWriter =tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY+"/testAccuracyWriter")
tf_summary_accuracy_log_var = tf.Variable(0.0)
tf.summary.scalar("Accuracy (Test/Train)", tf_summary_accuracy_log_var)
tfSummaryAccuracyMergedWriter = tf.summary.merge_all()

trainingTimeWriter = tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY+"/trainingTimeWriter")
testTimeWriter =tf.summary.FileWriter(LOG_DIR_FOR_TF_SUMMARY+"/testTimeWriter")
tf_summary_time_log_var = tf.Variable(0.0)
tf.summary.scalar("Time (Test/Train)", tf_summary_time_log_var)
tfSummaryTimeMergedWriter = tf.summary.merge_all()



##
## DATA CONSTANTS
##
# 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
SOUND_RECORD_SAMPLING_RATE=22050
TRACK_LENGTH=4*SOUND_RECORD_SAMPLING_RATE
# EVERY 4 second RECORD WILL BE CUT INTO 20 SLICES ( SO INPUT_SIZE WILL BE 4*22050/20 = 22050/5 = 4410 )
# 1 TRACK IS LIKE SUCCESIVE 20 SLICES (for LSTM)
NUMBER_OF_TIME_SLICES=4
# 10 types of sounds exist (car horn, ambulence, street music, children playing ...)
NUMBER_OF_CLASSES=10


##
## GENERAL CONSTANTS
##
OUTPUT_SIZE=NUMBER_OF_CLASSES
INPUT_SIZE_X=NUMBER_OF_TIME_SLICES
SLIDE_STEP=int(TRACK_LENGTH/NUMBER_OF_TIME_SLICES/2)
INPUT_SIZE_Y=int(TRACK_LENGTH/NUMBER_OF_TIME_SLICES+NUMBER_OF_TIME_SLICES*SLIDE_STEP)

##
## FULLY CONNECTED LAYER PARAMETERS
##
DROP_OUT=0.5
NUMBER_OF_FULLY_CONNECTED_NEURONS=512

##
## CNN PARAMETERS
##
CNN_KERNEL_COUNTS   = np.array([64   ,64  ,64   ,32    ,32    ,32    ,32 ,32 ,32 ,32 ,32 ])
CNN_KERNEL_X_SIZES     = np.array([2  ,1   ,1    ,1     ,1     ,1    ,1  ,1  ,1  ,1  ,1  ]) 
CNN_KERNEL_Y_SIZES     = np.array([7  , 7  , 5  , 5     ,3     ,3    ,3  ,3  ,3  ,2  ,2  ])
CNN_STRIDE_X_SIZES       = np.array([1    ,1   ,1    ,1     ,1    ,1   ,1  ,1  ,1  ,1  ,1   ])
CNN_STRIDE_Y_SIZES        = np.array([5    ,5   ,3    ,3     ,2    ,2   ,2  ,2  ,1  ,1  ,1  ])
CNN_POOL_X_SIZES       = np.array([1    ,1   ,1    ,1     ,1    ,1   ,1  ,1  ,1  ,1  ,1   ])
CNN_POOL_Y_SIZES          = np.array([5    ,5   ,3    ,3     ,2    ,2   ,2  ,2  ,1  ,1  ,1   ])

##
## TRAINING PARAMETERS
##
#LEARNING_RATE = 0.00001
LEARNING_RATE = 0.001
TRAINING_ITERATIONS=2000
#MINI_BATCH_SIZE=175
MINI_BATCH_SIZE=2

##
## LSTM PARAMETERS
##
NUMBER_OF_LSTM_LAYERS=2
LSTM_STATE_SIZE=256


##
## GLOBAL VARIABLES
##
EPSILON = 1e-4
MAX_VALUE_FOR_NORMALIZATION=0
MIN_VALUE_FOR_NORMALIZATION=0
fold_data_dictionary=dict()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
