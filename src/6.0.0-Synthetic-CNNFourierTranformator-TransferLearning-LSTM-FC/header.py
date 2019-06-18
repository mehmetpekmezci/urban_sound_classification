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
K           = importlib.import_module("keras")
sequence    = importlib.import_module("keras.preprocessing.sequence")
Sequential  = importlib.import_module("keras.models.Sequential")
Model       = importlib.import_module("keras.models.Model")
Input       = importlib.import_module("keras.layers.Input")
LSTM        = importlib.import_module("keras.layers.LSTM")
Dropout     = importlib.import_module("keras.layers.Dropout")
Dense       = importlib.import_module("keras.layers.Dense")

from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K



SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
uscLogger   = USCLogger(SCRIPT_DIR)
tfSummary   = TFSummary(SCRIPT_DIR)
dataObject  = Data(uscLogger.logger,SCRIPT_DIR)
SAVE_DIR    = SCRIPT_DIR+"/../../save/"+uscLogger.SCRIPT_NAME
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
##
## GENERAL CONSTANTS
##

FULLY_CONNECTED_LAYER=2048
#NUMBER_OF_INNER_HAIR_CELLS=3500
#NUMBER_OF_AFFERENT_NERVES_PER_INNER_HAIR_CELL=10
FOURIER_CNN_NUMBER_OF_LAYERS=3
FOURIER_CNN_NUMBER_OF_KERNELS=128
FOURIER_CNN_KERNEL_SIZE=128
FOURIER_CNN_STRIDE_SIZE=1
FOURIER_CNN_POOL_SIZE=1
AUTOENCODER_TRAINING_ITERATIONS=100
##
## TRAINING PARAMETERS
##
LEARNING_RATE = 0.001
TRAINING_ITERATIONS=9000
MINI_BATCH_SIZE=50
##
## LSTM PARAMETERS
##
NUMBER_OF_LSTM_LAYERS=1
LSTM_STATE_SIZE=1024
LSTM_FORGET_BIAS=0.5
##
## GLOBAL VARIABLES
##
EPSILON = 1e-4
