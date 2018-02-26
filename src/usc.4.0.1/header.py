#!/usr/bin/env python3
import importlib
math = importlib.import_module("math")
tf = importlib.import_module("tensorflow")
urllib3 = importlib.import_module("urllib3")
tarfile = importlib.import_module("tarfile")
csv = importlib.import_module("csv")
glob = importlib.import_module("glob")
sys = importlib.import_module("sys")
os = importlib.import_module("os")
argparse = importlib.import_module("argparse")
tempfile = importlib.import_module("tempfile")
np = importlib.import_module("numpy")
librosa = importlib.import_module("librosa")
pd = importlib.import_module("pandas")
time = importlib.import_module("time")
random = importlib.import_module("random")
plt = importlib.import_module("matplotlib.pyplot")
plt.style.use('ggplot'); plt.rcParams['font.family'] = 'serif'; plt.rcParams['font.serif'] = 'Ubuntu'; plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12; plt.rcParams['axes.labelsize'] = 11; plt.rcParams['axes.labelweight'] = 'bold'; plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9; plt.rcParams['ytick.labelsize'] = 9; plt.rcParams['legend.fontsize'] = 11; plt.rcParams['figure.titlesize'] = 13


fold_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
fold_dirs = ['fold1','fold10']



script_dir=os.path.dirname(os.path.realpath(__file__))
main_data_dir = script_dir+'/../../data/'
raw_data_dir = main_data_dir+'/0.raw/UrbanSound8K/audio'
csv_data_dir=main_data_dir+"/1.csv"
fold_data_dictionary=dict()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
# 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
# EVERY 4 second RECORD IS CUT INTO 20 SLICES ( SO LSTM_INPUT_SIZE WILL BE 4*22050/20 = 22050/5 = 4410
NUMBER_OF_TIME_SLICES=20
SOUND_RECORD_SAMPLING_RATE=22050
NUMBER_OF_CLASSES=10
LSTM_INPUT_SIZE=4*SOUND_RECORD_SAMPLING_RATE/NUMBER_OF_TIME_SLICES
LSTM_OUTPUT_SIZE=10
LSTM_NUMBER_OF_LAYERS=2
LSTM_SIZE=256
MAX_VALUE_FOR_NORMALIZATION=0
MIN_VALUE_FOR_NORMALIZATION=0
TRAINING_ITERATIONS=2000
MINI_BATCH_SIZE=10

