#!/usr/bin/env python3
from header import *

class Data :

 def __init__(self,logger,self.SCRIPT_DIR):
   self.logger=logger
   self.SCRIPT_DIR=SCRIPT_DIR
   self.FOLD_DIRS=['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
   self.MAIN_DATA_DIR=self.SCRIPT_DIR+'/../../data/'
   self.RAW_DATA_DIR=self.MAIN_DATA_DIR+'/0.raw/UrbanSound8K/audio'
   self.CSV_DATA_DIR=self.MAIN_DATA_DIR+"/1.csv"
   self.NP_DATA_DIR=self.MAIN_DATA_DIR+"/2.np"
   self.NUMBER_OF_CLASSES=10
   # 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
   self.SOUND_RECORD_SAMPLING_RATE=22050
   self.TRACK_LENGTH=4*self.SOUND_RECORD_SAMPLING_RATE
   self.TIME_SLICE_LENGTH=220 # input_size
   ## Humans can hear the voices between 100-20K Hz.
   ## 100Hz → 1/100 second → 22050 s.r / 100 = 220
   self.TIME_SLICE_OVERLAP_LENGTH=110 # half of the slice is overlapping with the nex time slice.
   self.NUMBER_OF_TIME_SLICES=int(int(self.TRACK_LENGTH-self.TIME_SLICE_LENGTH)/int(self.TIME_SLICE_LENGTH-self.TIME_SLICE_OVERLAP_LENGTH)+1)
   self.MAX_VALUE_FOR_NORMALIZATION=0
   self.MIN_VALUE_FOR_NORMALIZATION=0
   self.FOLD_DATA_DICTIONARY=dict()
   self.LAST_AUGMENTATION_CHOICE=0


 def parse_audio_files(self):
    sub4SecondSoundFilesCount=0
    for sub_dir in self.FOLD_DIRS:
      self.logger.info("Parsing : "+sub_dir)
      csvDataFile=open(self.CSV_DATA_DIR+"/"+sub_dir+".csv", 'w')
      csvDataWriter = csv.writer(csvDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      for file_path in glob.glob(os.path.join(self.RAW_DATA_DIR, sub_dir, '*.wav')):
         self.logger.info(file_path)
         try :
          classNumber=file_path.split('/')[-1].split('.')[0].split('-')[1]
          sound_data,sampling_rate=librosa.load(file_path)
          sound_data=np.array(sound_data)
          sound_data_duration=int(sound_data.shape[0]/self.SOUND_RECORD_SAMPLING_RATE)
          if sound_data_duration < 4 :
             sub4SecondSoundFilesCount=sub4SecondSoundFilesCount+1
             sound_data_in_4_second=np.zeros(4*self.SOUND_RECORD_SAMPLING_RATE)
             for i in range(sound_data.shape[0]):
               sound_data_in_4_second[i]=sound_data[i]
          else  :  
             sound_data_in_4_second=sound_data[:4*self.SOUND_RECORD_SAMPLING_RATE]
          sound_data_in_4_second=np.append(sound_data_in_4_second,[classNumber])
          csvDataWriter.writerow(sound_data_in_4_second)       
         except :
                e = sys.exc_info()[0]
                self.logger.info ("Exception :")
                self.logger.info (e)
      csvDataFile.close()       
      self.logger.info("sub4SecondSoundFilesCount="+str(sub4SecondSoundFilesCount));  

 def prepareData(self):
    self.logger.info("Starting to prepare the data ...  ")
    if not  os.path.exists(self.RAW_DATA_DIR) :
       if not  os.path.exists(self.MAIN_DATA_DIR+'/../data/0.raw'):
         os.makedirs(self.MAIN_DATA_DIR+'/../data/0.raw')   
    if not os.path.exists(self.MAIN_DATA_DIR+"/0.raw/UrbanSound8K"):
      if os.path.exists(self.MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz"):
         self.logger.info("Extracting "+self.MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz")
         tar = tarfile.open(self.MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz")
         tar.extractall(self.MAIN_DATA_DIR+'/../data/0.raw')
         tar.close()
         self.logger.info("Extracted "+self.MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz")
      else :   
         self.logger.info("download "+self.MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz from http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2  using firefox browser or chromium  and re-run this script")
         exit(1)
    if not os.path.exists(self.CSV_DATA_DIR) :
       os.makedirs(self.CSV_DATA_DIR)  
       self.parse_audio_files()
    if not os.path.exists(self.NP_DATA_DIR) :
       os.makedirs(self.NP_DATA_DIR)  
       self.save_as_np()
    self.logger.info("Data is READY  in CSV format. ")
    
 def save_as_np(self):
   self.logger.info ("save_as_np function started ...")
   fold_data_dictionary=dict()
   MAX_VALUE_FOR_NORMALIZATION=0
   MIN_VALUE_FOR_NORMALIZATION=0

   for fold in self.FOLD_DIRS:
     fold_data_dictionary[fold]=np.array(np.loadtxt(open(self.CSV_DATA_DIR+"/"+fold+".csv", "rb"), delimiter=","))
     for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*self.SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*self.SOUND_RECORD_SAMPLING_RATE]
          maxOfArray=np.amax(loadedDataX)
          minOfArray=np.amin(loadedDataX)
          if MAX_VALUE_FOR_NORMALIZATION < maxOfArray :
              MAX_VALUE_FOR_NORMALIZATION = maxOfArray
          if MIN_VALUE_FOR_NORMALIZATION > minOfArray :
              MIN_VALUE_FOR_NORMALIZATION = minOfArray
          ## Then append Y data to the end of row
        
     np.save(self.MAIN_DATA_DIR+"/2.np/"+fold+".npy",  fold_data_dictionary[fold]) 
     
   np.save(self.MAIN_DATA_DIR+"/2.np/minmax.npy",[MIN_VALUE_FOR_NORMALIZATION,MAX_VALUE_FOR_NORMALIZATION]) 
   self.logger.info ("save_as_np function finished ...")

 def normalize(self,data,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION):
    data_normalized=(data-MIN_VALUE_FOR_NORMALIZATION)/(MAX_VALUE_FOR_NORMALIZATION-MIN_VALUE_FOR_NORMALIZATION)
    return data_normalized

 def one_hot_encode_array(self,arrayOfYData):
   returnMatrix=np.empty([0,self.NUMBER_OF_CLASSES]);
   for i in range(arrayOfYData.shape[0]):
        one_hot_encoded_class_number = np.zeros(self.NUMBER_OF_CLASSES)
        one_hot_encoded_class_number[int(arrayOfYData[i])]=1
        returnMatrix=np.row_stack([returnMatrix, one_hot_encoded_class_number])
   return returnMatrix

 def one_hot_encode(self,classNumber):
   one_hot_encoded_class_number = np.zeros(self.NUMBER_OF_CLASSES)
   one_hot_encoded_class_number[int(classNumber)]=1
   return one_hot_encoded_class_number

 def load_all_np_data_back_to_memory(self,fold_data_dictionary):
   
   self.logger.info ("load_all_np_data_back_to_memory function started ...")
   for fold in self.FOLD_DIRS:
       self.logger.info ("loading from "+self.MAIN_DATA_DIR+"/2.np/"+fold+".npy  ...")
       fold_data_dictionary[fold]=np.load(self.MAIN_DATA_DIR+"/2.np/"+fold+".npy")
   minmax=np.load(self.MAIN_DATA_DIR+"/2.np/minmax.npy")
   MIN_VALUE_FOR_NORMALIZATION=minmax[0]
   MAX_VALUE_FOR_NORMALIZATION=minmax[1]

   self.logger.info ("load_all_np_data_back_to_memory function finished ...")
   return MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION
   

 def normalize_all_data(self,fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION):
     self.logger.info ("normalize_all_data function started ...")
     for fold in self.FOLD_DIRS:
       for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*self.SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*self.SOUND_RECORD_SAMPLING_RATE]
          normalizedLoadedDataX=self.normalize(loadedDataX,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION)
          fold_data_dictionary[fold][i]=np.append(normalizedLoadedDataX,loadedDataY)
     self.logger.info ("normalize_all_data function finished ...")
     return fold_data_dictionary

 def get_fold_data(self,fold,fold_data_dictionary):
     return np.random.permutation(fold_data_dictionary[fold])


 def slice_data(self,data,self.NUMBER_OF_TIME_SLICES) :
   return np.reshape(data,[-1,self.NUMBER_OF_TIME_SLICES,int(data.shape[1]/self.NUMBER_OF_TIME_SLICES)])
  

 def augment_speedx(self,sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    result=np.zeros(len(sound_array))
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    result_calculated= sound_array[ indices.astype(int) ]
    if len(result) > len(result_calculated) :
       result[:len(result_calculated)]=result_calculated
    else :
        result=result_calculated[:len(result)]
    return result

 def augment_volume(self,sound_array,factor):
    return factor * sound_array

 def augment_translate(self,snd_array, n):
    """ Translates the sound wave by n indices, fill the first n elements of the array with zeros """
    new_array=np.zeros(len(snd_array))
    new_array[n:]=snd_array[:-n]
    return new_array


 def augment_random(self,x_data):
  augmented_data= np.zeros([x_data.shape[0],x_data.shape[1]],np.float32)
  for i in range(x_data.shape[0]) :
    self.LAST_AUGMENTATION_CHOICE=(LAST_AUGMENTATION_CHOICE+1)%20
    augmented_data[i]=x_data[i]
    # 10 percent of being not augmented , if equals 0, then not augment, return directly real value
    if LAST_AUGMENTATION_CHOICE%10 != 0 :
      SPEED_FACTOR=0.8+LAST_AUGMENTATION_CHOICE/50
      TRANSLATION_FACTOR=int(5000*LAST_AUGMENTATION_CHOICE/10)
      INVERSE_FACTOR=LAST_AUGMENTATION_CHOICE%2
      if INVERSE_FACTOR == 1 :
       augmented_data[i]=-augmented_data[i]
      augmented_data[i]=self.augment_speedx(augmented_data[i],SPEED_FACTOR)
      augmented_data[i]=self.augment_translate(augmented_data[i],TRANSLATION_FACTOR)
      #augmented_data[i]=self.augment_volume(augmented_data[i],VOLUME_FACTOR)
  
  return augmented_data


 def convertToOverlappingSequentialData(self,x_input):
   self.logger.info("self.x_input.shape="+str(self.x_input.shape))
   stride=self.TIME_SLICE_LENGTH-TIME_SLICE_OVERLAP_LENGTH
   reshaped = tf.reshape(self.x_input, [MINI_BATCH_SIZE,1,-1,1])
   ones = tf.ones(self.TIME_SLICE_LENGTH, dtype=tf.float32)
   ident = tf.diag(ones)
   filter_dim = [1, self.TIME_SLICE_LENGTH, self.TIME_SLICE_LENGTH, 1]
   filter_matrix = tf.reshape(ident, filter_dim)
   stride_window = [1, 1, stride, 1]
   filtered_conv = []
   for f in tf.unstack(filter_matrix, axis=1):
      reshaped_filter = tf.reshape(f, [1, self.TIME_SLICE_LENGTH, 1, 1])
      c = tf.nn.conv2d(reshaped, reshaped_filter, stride_window, padding='VALID')
      filtered_conv.append(c)
   t = tf.stack(filtered_conv, axis=3)
   self.x_input_reshaped = tf.squeeze(t)
   self.logger.info("self.number_of_time_slices="+str(self.NUMBER_OF_TIME_SLICES))
   x_input_list = tf.unstack(self.x_input_reshaped,self.NUMBER_OF_TIME_SLICES, 1)
   self.logger.info("self.x_input[0].shape"+str(self.x_input_list[0].shape))
   return x_input_list

 def generate_single_synthetic_sample(self,MAX_NUMBER_OF_SYNTHETIC_FREQUENCY_PER_SAMPLE,self.SOUND_RECORD_SAMPLING_RATE,DURATION,MAX_HEARING_FREQUENCY):
    generated_data=np.zeros(DURATION*self.SOUND_RECORD_SAMPLING_RATE,np.float32)
    randomValue=np.random.rand()
    number_of_frequencies=int(randomValue*MAX_NUMBER_OF_SYNTHETIC_FREQUENCY_PER_SAMPLE)
    for i in range(number_of_frequencies):
      randomValue=np.random.rand()
      frequency=randomValue*MAX_HEARING_FREQUENCY # this generates 0-10000 float number,  from uniform dist.
      duration=randomValue*DURATION # this generates 0-4 float number,  from uniform dist.
      volume=randomValue*5
      sine_cosine_choice=int(randomValue*2)
      frequency_data=2*np.pi*np.arange(self.SOUND_RECORD_SAMPLING_RATE*duration)*frequency/self.SOUND_RECORD_SAMPLING_RATE
      if sine_cosine_choice == 0 :
          wave_data = (np.sin(frequency_data)).astype(np.float32)
      else :
          wave_data = (np.cos(frequency_data)).astype(np.float32)
      current_frequency_data=volume*wave_data
      start_point=generated_data.shape[0]-current_frequency_data.shape[0]
      #self.logger.info("Start point of this frequency within the sample :"+str(start_point)+")")
      start_point=int(randomValue*start_point)
      #self.logger.info("Start point of this frequency within the sample :"+str(start_point)+")")
      generated_data[start_point:start_point+current_frequency_data.shape[0]]+=current_frequency_data
    #self.play_sound(generated_data)
    #logger.info("Generated Data Length :"+str(generated_data.shape[0])+")")
    return generated_data

 def generate_normalized_synthetic_samples(self,fold):
    GENERATED_DATA=dict()
    if fold not in GENERATED_DATA :
      if os.path.exists(self.MAIN_DATA_DIR+"/2.np/generated_data-"+fold+".npy"):
        self.logger.info("Loading Already Generated Synthetic Sound Sample Data from "+self.MAIN_DATA_DIR+"/2.np/generated_data-"+fold+".npy")
        GENERATED_DATA[fold]=np.load(self.MAIN_DATA_DIR+"/2.np/generated_data-"+fold+".npy")
      else :
        self.logger.info("Starting to Generate Synthetic Sound Sample Data for fold "+str(fold))
        global NUMBER_OF_SYNTHETIC_TRAINNG_SAMPLES,MAX_NUMBER_OF_SYNTHETIC_FREQUENCY_PER_SAMPLE,self.SOUND_RECORD_SAMPLING_RATE,DURATION,MAX_HEARING_FREQUENCY
        samples=np.zeros((NUMBER_OF_SYNTHETIC_TRAINNG_SAMPLES,DURATION*self.SOUND_RECORD_SAMPLING_RATE),np.float32)
        for i in range(NUMBER_OF_SYNTHETIC_TRAINNG_SAMPLES):
          samples[i]=self.generate_single_synthetic_sample(MAX_NUMBER_OF_SYNTHETIC_FREQUENCY_PER_SAMPLE,self.SOUND_RECORD_SAMPLING_RATE,DURATION,MAX_HEARING_FREQUENCY)
        max_value=np.amax(samples)
        min_value=np.amin(samples)
        samples=normalize(samples,max_value,min_value)
        GENERATED_DATA[fold]=samples
        self.logger.info("Saving Generated Synthetic Sound Sample Data to "+self.MAIN_DATA_DIR+"/2.np/generated_data-"+fold+".npy")
        np.save(self.MAIN_DATA_DIR+"/2.np/generated_data-"+fold+".npy", samples)
        self.logger.info("Finished to Generate Synthetic Sound Sample Data for fold "+str(fold))
    return  GENERATED_DATA[fold]

 def play_sound(self,sound_data):
  global self.SOUND_RECORD_SAMPLING_RATE
  self.logger.info("sound_data.shape="+str(sound_data.shape))
  self.logger.info("SOUND_RECORD_SAMPLING_RATE="+str(self.SOUND_RECORD_SAMPLING_RATE))
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.SOUND_RECORD_SAMPLING_RATE, output=True)
  stream.write(sound_data[:22050],self.SOUND_RECORD_SAMPLING_RATE)
  stream.write(sound_data[22050:44100],self.SOUND_RECORD_SAMPLING_RATE)
  stream.write(sound_data[44100:66150],self.SOUND_RECORD_SAMPLING_RATE)
  stream.write(sound_data[66150:88200],self.SOUND_RECORD_SAMPLING_RATE)
  stream.stop_stream()
  stream.close()
  p.terminate()
  self.logger.info("Finished To Play Sound")
# generate samples, note conversion to float32 array
# for paFloat32 sample values must be in range [-1.0, 1.0]
### NOTE:  3500 Inner Hair Cell, each connected to ~10 neurons, they connect to auditory nucleus, then signals are transferred to the auditory cortex1 then to cortex2
###        Humans can hear 20Hz to 20 000Hz
###        Human  voice frq : 100 to 10000 Hz
###        Human  talk voice frq : 100 to 8000 Hz


