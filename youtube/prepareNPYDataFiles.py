#!/usr/bin/env python3
##
## IMPORTS
##
import importlib
glob        = importlib.import_module("glob")
os          = importlib.import_module("os")
sys         = importlib.import_module("sys")
np          = importlib.import_module("numpy")
librosa     = importlib.import_module("librosa")
random      = importlib.import_module("random")


# 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
SOUND_RECORD_SAMPLING_RATE=22050
TRACK_LENGTH=4*SOUND_RECORD_SAMPLING_RATE
NUMBER_OF_RECORDS_PER_NPY_FILE=100

for category in glob.glob('raw/*/'):
   print('Importing ',category,' ...')
   counter=0
   sound_data_group=[] ## sound data is shuffled and grouped by NUMBER_OF_RECORDS_PER_NPY_FILE records
   fileList=glob.glob(category+'/*.wav')
        
   for wav_file in random.sample(fileList,len(fileList)):
#     try :
          print('Processing File: ',wav_file)
          sound_data,sampling_rate=librosa.load(wav_file)
          sound_data=np.array(sound_data)

          if sound_data.shape[0] < TRACK_LENGTH :
             sound_data_in_4_second=np.zeros(4*SOUND_RECORD_SAMPLING_RATE)
             sound_data_in_4_second[:sound_data.shape[0]]=sound_data
          else  :
             sound_data_in_4_second=sound_data[:4*SOUND_RECORD_SAMPLING_RATE]

          sound_data_group.append(sound_data_in_4_second)

          print('len(sound_data_group)= ',len(sound_data_group))

          if len(fileList) >=  NUMBER_OF_RECORDS_PER_NPY_FILE :
            if len(sound_data_group) == NUMBER_OF_RECORDS_PER_NPY_FILE :
               print('Writing Data File No: ',counter)
               np.save(category+'/data.'+str(counter)+'.npy',sound_data_group) 
               sound_data_group=[]
               counter+=1
            else :
               if len(sound_data_group)+counter*NUMBER_OF_RECORDS_PER_NPY_FILE == len(fileList) :
                 print('Writing Data File No: ',counter)
                 np.save(category+'/data.'+str(counter)+'.npy',sound_data_group) 
                 sound_data_group=[]
                 counter+=1
          else :
             if len(sound_data_group) == len(fileList) :
               print('Writing Data File No: ',counter)
               np.save(category+'/data.'+str(counter)+'.npy',sound_data_group) 
               sound_data_group=[]
               counter+=1
#     except :
#          e = sys.exc_info()[0]

