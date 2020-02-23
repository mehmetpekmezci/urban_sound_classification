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

##
## DATA DIRECTORY NAMES
##
FOLD_DIRS = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
#FOLD_DIRS = ['fold1','fold10']
FOLD_DIRS = []
SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME=os.path.basename(SCRIPT_DIR)
MAIN_DATA_DIR = SCRIPT_DIR+'/../../data/'
RAW_DATA_DIR = MAIN_DATA_DIR+'/0.raw/UrbanSound8K/audio'
CSV_DATA_DIR=MAIN_DATA_DIR+"/1.csv"
NP_DATA_DIR=MAIN_DATA_DIR+"/2.np"
LOG_DIR_FOR_LOGGER=SCRIPT_DIR+"/../../logs/logger/"+SCRIPT_NAME
LOG_DIR_FOR_TF_SUMMARY=SCRIPT_DIR+"/../../logs/tf-summary/"+SCRIPT_NAME+"/"+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))
SAVE_DIR=SCRIPT_DIR+"/../../save/"+SCRIPT_NAME


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
DURATION=4
TRACK_LENGTH=DURATION*SOUND_RECORD_SAMPLING_RATE
# 10 types of sounds exist (car horn, ambulence, street music, children playing ...)
NUMBER_OF_CLASSES=10

MAX_NUMBER_OF_SYNTHETIC_FREQUENCY_PER_SAMPLE=4
MIN_HEARING_FREQUENCY=20
#MAX_HEARING_FREQUENCY=20000
MAX_HEARING_FREQUENCY=10000 #after 10000 to 20000 it is hardly heard.
###        Humans can hear 20Hz to 20 000Hz
###        Human  voice frq : 100 to 10000 Hz
###        Human  talk voice frq : 100 to 8000 Hz




##
## GENERAL CONSTANTS
##
OUTPUT_SIZE=NUMBER_OF_CLASSES
INPUT_SIZE=TRACK_LENGTH



##
## FULLY CONNECTED LAYER PARAMETERS
##
DROP_OUT=0.5
KEEP_PROB=DROP_OUT

RESIDUAL_ATTENTION_CNN_KERNEL_ENCODERS     = np.array([  0, 1, 1])
RESIDUAL_ATTENTION_CNN_KERNEL_COUNTS       = np.array([ 16,16,16])
RESIDUAL_ATTENTION_CNN_KERNEL_X_SIZES      = np.array([  1, 1, 1])
RESIDUAL_ATTENTION_CNN_KERNEL_Y_SIZES      = np.array([ 64, 8,64])
RESIDUAL_ATTENTION_CNN_STRIDE_X_SIZES      = np.array([  1, 1, 1])
RESIDUAL_ATTENTION_CNN_STRIDE_Y_SIZES      = np.array([  1, 1, 1])
RESIDUAL_ATTENTION_CNN_POOL_X_SIZES        = np.array([  1, 1, 1])
RESIDUAL_ATTENTION_CNN_POOL_Y_SIZES        = np.array([ 16, 1,16])




FOURIER_CNN_KERNEL_COUNTS       = np.array([ 16,32,64])
FOURIER_CNN_KERNEL_X_SIZES      = np.array([  1, 1, 1])
FOURIER_CNN_KERNEL_Y_SIZES      = np.array([ 64,32,16])
FOURIER_CNN_STRIDE_X_SIZES      = np.array([  1, 1, 1])
FOURIER_CNN_STRIDE_Y_SIZES      = np.array([  1, 1, 1])
FOURIER_CNN_POOL_X_SIZES        = np.array([  1, 1, 1])
FOURIER_CNN_POOL_Y_SIZES        = np.array([ 16, 8, 4])

FULLY_CONNECTED_LAYERS=[256]


##
## CNN PARAMETERS
##
## AUDIO DATA IS ONE DIMENSIONAL  ( that is why *x* is 1)
#CNN_KERNEL_COUNTS       = np.array([128,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32])
#CNN_KERNEL_X_SIZES      = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#CNN_KERNEL_Y_SIZES      = np.array([ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
#CNN_STRIDE_X_SIZES      = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#CNN_STRIDE_Y_SIZES      = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#CNN_POOL_X_SIZES        = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#CNN_POOL_Y_SIZES        = np.array([ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

CNN_KERNEL_COUNTS       = np.array([ 16,16,16,16,16,16,16])
CNN_KERNEL_X_SIZES      = np.array([  1, 1, 1, 1, 1, 1, 1])
CNN_KERNEL_Y_SIZES      = np.array([  4, 4, 4, 4, 4, 4, 4])
CNN_STRIDE_X_SIZES      = np.array([  1, 1, 1, 1, 1, 1, 1])
CNN_STRIDE_Y_SIZES      = np.array([  1, 1, 1, 1, 1, 1, 1])
CNN_POOL_X_SIZES        = np.array([  1, 1, 1, 1, 1, 1, 1])
CNN_POOL_Y_SIZES        = np.array([  2, 2, 2, 2, 2, 1, 1])

METRIC_CNN_KERNEL_COUNTS       = np.array([ 16,16,16])
METRIC_CNN_KERNEL_X_SIZES      = np.array([  1, 1, 1])
METRIC_CNN_KERNEL_Y_SIZES      = np.array([  4, 4, 4])
METRIC_CNN_STRIDE_X_SIZES      = np.array([  1, 1, 1])
METRIC_CNN_STRIDE_Y_SIZES      = np.array([  1, 1, 1])
METRIC_CNN_POOL_X_SIZES        = np.array([  1, 1, 1])
METRIC_CNN_POOL_Y_SIZES        = np.array([  1, 1, 1])



##
## TRAINING PARAMETERS
##
#LEARNING_RATE = 0.00001
#LEARNING_RATE = 0.000001
LEARNING_RATE = 0.00001
LEARNING_RATE_BETA1 = 0.9
LEARNING_RATE_BETA2 = 0.999

LOSS_WEIGHT_1=1/3
LOSS_WEIGHT_2=1/3
LOSS_WEIGHT_3=1/3


TRAINING_ITERATIONS=9999
MINI_BATCH_SIZE=10

##
## GLOBAL VARIABLES
##
EPSILON = 1e-4
MAX_VALUE_FOR_NORMALIZATION=0
MIN_VALUE_FOR_NORMALIZATION=0
fold_data_dictionary=dict()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#sess = tf.InteractiveSession(config=config)

LAST_AUGMENTATION_CHOICE=0

