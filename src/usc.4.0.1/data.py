#!/usr/bin/env python3
from header import *

def parse_audio_files():

    global raw_data_dir,csv_data_dir,fold_dirs
    sub4SecondSoundFilesCount=0
    for sub_dir in fold_dirs:
      print("Parsing : "+sub_dir)
      csvDataFile=open(csv_data_dir+"/"+sub_dir+".csv", 'w')
      csvDataWriter = csv.writer(csvDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      for file_path in glob.glob(os.path.join(raw_data_dir, sub_dir, '*.wav')):
         print(file_path)
         try :
          classNumber=file_path.split('/')[-1].split('.')[0].split('-')[1]
          sound_data,sampling_rate=librosa.load(file_path)
          sound_data=np.array(sound_data)
          sound_data_duration=int(sound_data.shape[0]/SOUND_RECORD_SAMPLING_RATE)
          if sound_data_duration < 4 :
             sub4SecondSoundFilesCount=sub4SecondSoundFilesCount+1
             sound_data_in_4_second=np.zeros(4*SOUND_RECORD_SAMPLING_RATE)
             for i in range(sound_data.shape[0]):
               sound_data_in_4_second[i]=sound_data[i]
          else  :  
             sound_data_in_4_second=sound_data[:4*SOUND_RECORD_SAMPLING_RATE]
          sound_data_in_4_second=np.append(sound_data_in_4_second,[classNumber])
          csvDataWriter.writerow(sound_data_in_4_second)       
         except :
                e = sys.exc_info()[0]
                print ("Exception :")
                print (e)
      csvDataFile.close()       
    print("sub4SecondSoundFilesCount="+str(sub4SecondSoundFilesCount));  


def prepareData():
    print ("prepareData function ...")
    if not  os.path.exists(raw_data_dir) :
       if not  os.path.exists(main_data_dir+'/../data/0.raw'):
         os.makedirs(main_data_dir+'/../data/0.raw')   
    if not os.path.exists(main_data_dir+"/0.raw/UrbanSound8K"):
      if os.path.exists(main_data_dir+"/0.raw/UrbanSound8K.tar.gz"):
         print("Extracting "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar = tarfile.open(main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar.extractall(main_data_dir+'/../data/0.raw')
         tar.close()
         print("Extracted "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
      else :   
        print("download "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz from http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2  using firefox browser or chromium  and re-run this script")
         #print("download "+main_data_dir+"/0.raw/UrbanSound8K.tar.gz from https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound8k.html using firefox browser or chromium  and re-run this script")
        exit(1)
#         http = urllib3.PoolManager()
#         chunk_size=100000
#         r = http.request('GET', 'http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2', preload_content=False)
#         with open(main_data_dir+"/0.raw/UrbanSound8K.tar.gz", 'wb') as out:
#          while True:
#           data = r.read(chunk_size)
#           if not data:
#             break
#           out.write(data)
#         r.release_conn()

    if not os.path.exists(csv_data_dir) :
       os.makedirs(csv_data_dir)  
       parse_audio_files()
    print ("prepareData function finished ...")

def normalize(data):
    global MAX_VALUE_FOR_NORMALIZATION , MIN_VALUE_FOR_NORMALIZATION
    data_normalized=(data-MIN_VALUE_FOR_NORMALIZATION)/(MAX_VALUE_FOR_NORMALIZATION-MIN_VALUE_FOR_NORMALIZATION)
    return data_normalized

def one_hot_encode_array(arrayOfYData):
   returnMatrix=np.empty([0,NUMBER_OF_CLASSES]);
   for i in range(arrayOfYData.shape[0]):
        one_hot_encoded_class_number = np.zeros(NUMBER_OF_CLASSES)
        one_hot_encoded_class_number[int(arrayOfYData[i])]=1
        returnMatrix=np.row_stack([returnMatrix, one_hot_encoded_class_number])
   return returnMatrix

def one_hot_encode(classNumber):
   one_hot_encoded_class_number = np.zeros(NUMBER_OF_CLASSES)
   one_hot_encoded_class_number[int(classNumber)]=1
   return one_hot_encoded_class_number

def load_all_csv_data_back_to_memory():
     print ("load_all_csv_data_back_to_memory function started ...")
     global fold_data_dictionary, MAX_VALUE_FOR_NORMALIZATION ,  MIN_VALUE_FOR_NORMALIZATION
     for fold in fold_dirs:
       fold_data_dictionary[fold]=np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
       for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*SOUND_RECORD_SAMPLING_RATE]
          maxOfArray=np.amax(loadedDataX)
          minOfArray=np.amin(loadedDataX)
          if MAX_VALUE_FOR_NORMALIZATION < maxOfArray :
              MAX_VALUE_FOR_NORMALIZATION = maxOfArray
          if MIN_VALUE_FOR_NORMALIZATION > minOfArray :
              MIN_VALUE_FOR_NORMALIZATION = minOfArray
          ## Then append Y data to the end of row
          fold_data_dictionary[fold][i]=np.append(loadedDataX,loadedDataY)
     print ("load_all_csv_data_back_to_memory function finished ...")

def normalize_all_data():
     print ("normalize_all_data function started ...")
     global fold_data_dictionary
     for fold in fold_dirs:
       for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*SOUND_RECORD_SAMPLING_RATE]
          normalizedLoadedDataX=normalize(loadedDataX)
          fold_data_dictionary[fold][i]=np.append(normalizedLoadedDataX,loadedDataY)
     print ("normalize_all_data function finished ...")

def get_fold_data(fold):
     global fold_data_dictionary
     return np.random.permutation(fold_data_dictionary[fold])


def slice_data(data,NUMBER_OF_SLICES) :
   NUMBER_OF_BATCHES=data.shape[0]
   SLICE_WIDTH=int(data.shape[1]/NUMBER_OF_SLICES)
   sliced_data=np.empty((NUMBER_OF_BATCHES,NUMBER_OF_SLICES,SLICE_WIDTH))
   for current_batch in range(NUMBER_OF_BATCHES) :
     for current_slice in range(NUMBER_OF_SLICES) :
         sliced_data[current_batch][current_slice]=data[current_batch][current_slice*SLICE_WIDTH:(current_slice+1)*SLICE_WIDTH]
   return sliced_data
                
def replicate_data(data,NUMBER_OF_SLICES) :
   NUMBER_OF_BATCHES=data.shape[0]
   replicated_data=np.empty((NUMBER_OF_BATCHES,NUMBER_OF_SLICES,data.shape[1]))
   for current_batch in range(NUMBER_OF_BATCHES) :
        for current_slice in range(NUMBER_OF_SLICES) :
           ## replicate all data (y_one_hot_encoded_batch) ,for all slices
           replicated_data[current_batch][current_slice]=data[current_batch]
   return replicated_data
                
  
