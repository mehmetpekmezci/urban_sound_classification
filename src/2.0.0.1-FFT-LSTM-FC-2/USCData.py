#!/usr/bin/env python3
from USCHeader import *

class USCData :
 def __init__(self, logger):
   self.logger  = logger
   self.script_dir=os.path.dirname(os.path.realpath(__file__))
   self.script_name=os.path.basename(self.script_dir)
   self.fold_dirs=['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
   #self.fold_dirs=['fold1','fold10']
   #self.fold_dirs=['fold1']
   self.main_data_dir=self.script_dir+'/../../data/'
   self.raw_data_dir=self.main_data_dir+'/0.raw/UrbanSound8K/audio'
   self.csv_data_dir=self.main_data_dir+'/1.csv'
   self.np_data_dir=self.main_data_dir+'/2.np'
   self.sound_record_sampling_rate=22050 # 22050 sample points per second
   self.track_length=4*self.sound_record_sampling_rate # 4 seconds record
   self.time_slice_length=440
   self.time_slice_overlap_length=265
   self.number_of_time_slices=math.ceil(self.track_length/(self.time_slice_length-self.time_slice_overlap_length))
   self.number_of_classes=10
   self.mini_batch_size=100
   self.fold_data_dictionary=dict()

 def parse_audio_files(self):
    sub4SecondSoundFilesCount=0
    for sub_dir in self.fold_dirs:
      self.logger.info("Parsing : "+sub_dir)
      csvDataFile=open(self.csv_data_dir+"/"+sub_dir+".csv", 'w')
      csvDataWriter = csv.writer(csvDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      for file_path in glob.glob(os.path.join(self.raw_data_dir, sub_dir, '*.wav')):
         self.logger.info(file_path)
         try :
          classNumber=file_path.split('/')[-1].split('.')[0].split('-')[1]
          sound_data,sampling_rate=librosa.load(file_path)
          sound_data=np.array(sound_data)
          sound_data_duration=int(sound_data.shape[0]/self.sound_record_sampling_rate)
          if sound_data_duration < 4 :
             sub4SecondSoundFilesCount=sub4SecondSoundFilesCount+1
             sound_data_in_4_second=np.zeros(4*self.sound_record_sampling_rate)
             for i in range(sound_data.shape[0]):
               sound_data_in_4_second[i]=sound_data[i]
          else  :  
             sound_data_in_4_second=sound_data[:4*self.sound_record_sampling_rate]
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
    if not  os.path.exists(self.raw_data_dir) :
       if not  os.path.exists(self.main_data_dir+'/../data/0.raw'):
         os.makedirs(self.main_data_dir+'/../data/0.raw')   
    if not os.path.exists(self.main_data_dir+"/0.raw/UrbanSound8K"):
      if os.path.exists(self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz"):
         self.logger.info("Extracting "+self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar = tarfile.open(self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
         tar.extractall(self.main_data_dir+'/../data/0.raw')
         tar.close()
         self.logger.info("Extracted "+self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz")
      else :   
         self.logger.info("download "+self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz from http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2  using firefox browser or chromium  and re-run this script")
         # self.logger.info("download "+self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz from https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound8k.html using firefox browser or chromium  and re-run this script")
         exit(1)
#         http = urllib3.PoolManager()
#         chunk_size=100000
#         r = http.request('GET', 'http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2', preload_content=False)
#         with open(self.main_data_dir+"/0.raw/UrbanSound8K.tar.gz", 'wb') as out:
#          while True:
#           data = r.read(chunk_size)
#           if not data:
#             break
#           out.write(data)
#         r.release_conn()

    if not os.path.exists(self.csv_data_dir) :
       os.makedirs(self.csv_data_dir)  
       parse_audio_files()
    if not os.path.exists(self.np_data_dir) :
       os.makedirs(self.np_data_dir)  
       save_as_np()
    self.logger.info("Data is READY  in CSV format. ")
    self.load_all_np_data_back_to_memory()

    
 def save_as_np(self):
    self.logger.info ("save_as_np function started ...")
    fold_data_dictionary=dict()
    max_value_for_normalization=0
    min_value_for_normalization=0
 
    for fold in self.fold_dirs:
      fold_data_dictionary[fold]=np.array(np.loadtxt(open(self.csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
      for i in range(fold_data_dictionary[fold].shape[0]) :
           loadedData=fold_data_dictionary[fold][i]
           loadedDataX=loadedData[:4*self.sound_record_sampling_rate]
           loadedDataY=loadedData[4*self.sound_record_sampling_rate]
           maxOfArray=np.amax(loadedDataX)
           minOfArray=np.amin(loadedDataX)
           if max_value_for_normalization < maxOfArray :
               max_value_for_normalization = maxOfArray
           if min_value_for_normalization > minOfArray :
               min_value_for_normalization = minOfArray
           ## Then append Y data to the end of row
         
      np.save(self.main_data_dir+"/2.np/"+fold+".npy",  fold_data_dictionary[fold]) 
      
    np.save(self.main_data_dir+"/2.np/minmax.npy",[min_value_for_normalization,max_value_for_normalization]) 
    self.logger.info ("save_as_np function finished ...")

 def normalize(self,data):
    normalized_data = data/np.linalg.norm(data) 
    return normalized_data

 def one_hot_encode_array(self,arrayOfYData):
    returnMatrix=np.empty([0,self.number_of_classes]);
    for i in range(arrayOfYData.shape[0]):
         one_hot_encoded_class_number = np.zeros(self.number_of_classes)
         one_hot_encoded_class_number[int(arrayOfYData[i])]=1
         returnMatrix=np.row_stack([returnMatrix, one_hot_encoded_class_number])
    return returnMatrix

 def one_hot_encode(self,classNumber):
    one_hot_encoded_class_number = np.zeros(self.number_of_classes)
    one_hot_encoded_class_number[int(classNumber)]=1
    return one_hot_encoded_class_number

 def load_all_np_data_back_to_memory(self):
   
    self.logger.info ("load_all_np_data_back_to_memory function started ...")
    for fold in self.fold_dirs:
        self.logger.info ("loading from "+self.main_data_dir+"/2.np/"+fold+".npy  ...")
        self.fold_data_dictionary[fold]=np.load(self.main_data_dir+"/2.np/"+fold+".npy")
    minmax=np.load(self.main_data_dir+"/2.np/minmax.npy")
    min_value_for_normalization=minmax[0]
    max_value_for_normalization=minmax[1]
 
    self.logger.info ("load_all_np_data_back_to_memory function finished ...")
    return max_value_for_normalization,min_value_for_normalization
   

 def get_fold_data(self,fold):
    return np.random.permutation(self.fold_data_dictionary[fold])

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

 def augment_inverse(self,sound_array):
    return -sound_array

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
       choice=np.random.rand()*20
       augmented_data[i]=x_data[i]
       # 10 percent of being not augmented , if equals 0, then not augment, return directly real value
       if choice%10 != 0 :
         SPEED_FACTOR=0.8+choice/50
         TRANSLATION_FACTOR=int(5000*choice/10)+1
         INVERSE_FACTOR=choice%2
         if INVERSE_FACTOR == 1 :
          augmented_data[i]=-augmented_data[i]
         augmented_data[i]=self.augment_speedx(augmented_data[i],SPEED_FACTOR)
         augmented_data[i]=self.augment_translate(augmented_data[i],TRANSLATION_FACTOR)
         #augmented_data[i]=self.augment_volume(augmented_data[i],VOLUME_FACTOR)
    return augmented_data

 def overlapping_hanning_slice(self,x_data):
    sliced_and_overlapped_data=np.zeros([self.mini_batch_size,self.number_of_time_slices,self.time_slice_length])
    step=self.time_slice_length-self.time_slice_overlap_length
    hanning_window=np.hanning(self.time_slice_length)
    for i in range(self.mini_batch_size):
        for j in range(self.number_of_time_slices):
            step_index=j*step
            if step_index+self.time_slice_length>x_data.shape[1]:
                overlapped_time_slice=np.zeros(self.time_slice_length)
                overlapped_time_slice[0:int(x_data.shape[1]-step_index)]=x_data[i,step_index:x_data.shape[1]]
            else :
                overlapped_time_slice=x_data[i,step_index:step_index+self.time_slice_length]
            sliced_and_overlapped_data[i,j]=overlapped_time_slice
            sliced_and_overlapped_data[i,j]*=hanning_window
    #self.logger.info(sliced_and_overlapped_data.shape)
    #self.logger.info(sliced_and_overlapped_data[0][100][step])
    #self.logger.info(sliced_and_overlapped_data[0][101][0])
    return sliced_and_overlapped_data
    #x_input_list = tf.unstack(self.x_input_reshaped, self.number_of_time_slices, 1)

 def fft(self,x_data):
    #deneme_data=x_data[15][25]
    #self.logger.info("deneme_datae[18]="+str(deneme_data[18]))
    #fft_deneme_data=np.abs(np.fft.fft(deneme_data))
    #self.logger.info("fft_deneme_data[18]="+str(fft_deneme_data[18]))
    x_data = np.abs(np.fft.fft(x_data))
    #self.logger.info("x_data[15][25][18]="+str(x_data[15][25][18]))
    return x_data

 def convert_to_list_for_parallel_lstms(self,x_data,num_of_paralel_lstms,lstm_time_steps):
     print(x_data.shape)
     x_data=np.reshape(x_data,(self.mini_batch_size,num_of_paralel_lstms,lstm_time_steps,self.time_slice_length))
     ## switch axes of batch_size and parallel_lstms, then convert it to list according to first axis. --> this will give us list of matrices of shape (mini_batch_size,lstm_time_steps,time_slice_lentgh)
     x_list=np.swapaxes(x_data,0,1).tolist()
     return x_list

