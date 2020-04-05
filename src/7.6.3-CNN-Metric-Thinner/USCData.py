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
   self.time_slice_length=2000
   #self.time_slice_length=440
   #self.time_slice_length=55
   self.time_slice_overlap_length=200
   #self.time_slice_overlap_length=265
   #self.time_slice_overlap_length=30
   self.number_of_time_slices=math.ceil(self.track_length/(self.time_slice_length-self.time_slice_overlap_length))
   self.number_of_classes=10
   self.mini_batch_size=20

   #self.mini_batch_size=40  # very slow learning
   self.fold_data_dictionary=dict()
   self.youtube_data_file_dictionary=dict()
   self.youtube_data_file_category_enumeration=dict()
   self.current_youtube_data=None
   self.youtube_data_max_category_data_file_count=0
   self.current_data_file_number=0
   self.prepareData()
   self.findListOfYoutubeDataFiles()
   self.youtubeDataLoaderThread=threading.Thread(target=self.youtube_data_loader_thread_worker_method, daemon=True)
   self.youtubeDataLoaderThread.start()
   

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
    # arrayOfYData.shape[0]==batch_size
    # all-zero for unknown class youtube data
    returnMatrix=np.zeros([arrayOfYData.shape[0],self.number_of_classes]);
    for i in range(arrayOfYData.shape[0]):
       classNumber=int(arrayOfYData[i])
       if classNumber<10 :
         returnMatrix[i,classNumber]=1
#       else :
#         let the row be all 0 (M.P.)
    return returnMatrix

 def similarity_array(self,arrayOfYData_1,arrayOfYData_2):
    indices=np.where(np.equal(arrayOfYData_1, arrayOfYData_2))[0]
    returnMatrix=np.zeros([arrayOfYData_1.shape[0]]);
    returnMatrix[indices]=1
    return returnMatrix

 def is_all_data_labeled(self,arrayOfYData):
    indices=np.where(arrayOfYData>=10)[1]
    if len(indices) > 0 :
      return 0
    return 1


 def one_hot_encode(self,classNumber):
    one_hot_encoded_class_number = np.zeros(self.number_of_classes)
    one_hot_encoded_class_number[int(classNumber)]=1
    return one_hot_encoded_class_number


 def findListOfYoutubeDataFiles(self):
    self.logger.info ("Crawling Youtube Data Files From Directory ../../youtube/downloads/ ...")
    if not os.path.exists('../../youtube/raw/'):
        self.logger.info("../../youtube/raw/ directory does not exist.")
        self.logger.info("Please do the following :")
        self.logger.info(" 1. cd ../../youtube/")
        self.logger.info(" 2. ./download.sh")
        self.logger.info(" 3. ./convertAll.sh")
        self.logger.info(" 4. ./splitAll.sh")
        self.logger.info(" 5. python3 prepareNPYDataFiles.py")
        exit(1);
    if len(glob.glob('../../youtube/raw/*/*.npy')) == 0:
        self.logger.info("../../youtube/raw/*/*.npy data files do not exist , first go to ../../youtube directory and run 'python3 prepareNPYDataFiles.py' ")
        exit(1);

    enum=100
    for category in glob.glob('../../youtube/raw/*/'):
      dataFileList=glob.glob(category+'/*.npy')
      if len(dataFileList) > self.youtube_data_max_category_data_file_count :
          self.youtube_data_max_category_data_file_count=len(dataFileList)
      self.youtube_data_file_dictionary[category]=random.sample(dataFileList,len(dataFileList))
      self.youtube_data_file_category_enumeration[category]=enum
      enum+=1
    self.logger.info("There are "+str(enum-100)+" categories of youtube data")


 def getNextYoutubeData(self):
     if self.current_youtube_data is None :
        self.logger.info("self.current_youtube_data is None , so first loading youtube data to memory")
        self.loadNextYoutubeData()
     returnValue=self.current_youtube_data
     self.current_youtube_data=None
     return returnValue

 def loadNextYoutubeData(self):
     local_youtube_data=np.empty([0,4*self.sound_record_sampling_rate+1])
     for category in  self.youtube_data_file_dictionary :
         dataFileList= self.youtube_data_file_dictionary[category]
         if len(dataFileList) > self.current_data_file_number :
             #self.logger.info("loading"+ category+'/data.'+str(self.current_data_file_number)+'.npy')
             loadedData=np.load(category+'/data.'+str(self.current_data_file_number)+'.npy')
             loadedData=loadedData[:,:4*self.sound_record_sampling_rate]
             newLoadedData=np.zeros((loadedData.shape[0],loadedData.shape[1]+1))
             newLoadedData[:,:-1]=loadedData
             loadedData=newLoadedData
             #SET out of range category of current data

             loadedData[:,4*self.sound_record_sampling_rate]=np.full((loadedData.shape[0]),self.youtube_data_file_category_enumeration[category])
             #listOf4SecondRecords=loadedData.tolist()
             #self.logger.info(len(listOf4SecondRecords))
             local_youtube_data=np.vstack((local_youtube_data,loadedData)) ## this appends listOf4SecondRecords to local_youtube_data
     self.current_data_file_number= (self.current_data_file_number+1)%self.youtube_data_max_category_data_file_count  
     np.random.shuffle(local_youtube_data)
     self.current_youtube_data=local_youtube_data
     #self.logger.info(self.current_youtube_data.shape)

 def youtube_data_loader_thread_worker_method(self):
     self.logger.info(" youtube_data_loader_thread_worker_method is called ")
     while(True):
        if self.current_youtube_data is None :
          self.loadNextYoutubeData()
        else :
          time.sleep(1)

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
    
 def augment_translate_and_set_zero_and_occlude(self,snd_array,TRANSLATION_FACTOR,ZERO_INDEX,OCCLUDE_START_INDEX,OCCLUDE_WIDTH):
    """ Translates the sound wave by n indices, fill the first n elements of the array with zeros """
    new_array=np.zeros(len(snd_array))
    new_array[TRANSLATION_FACTOR:OCCLUDE_START_INDEX]=snd_array[:(-TRANSLATION_FACTOR+OCCLUDE_START_INDEX)]
    new_array[OCCLUDE_START_INDEX+OCCLUDE_WIDTH:-ZERO_INDEX]=snd_array[OCCLUDE_START_INDEX+OCCLUDE_WIDTH:-ZERO_INDEX]
    return new_array

 def overlapping_slice(self,x_data,hanning=False):
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
            if hanning :
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

 def convert_to_list_of_word2vec_window_sized_data(self,x_data):
     #print(x_data.shape)
     result=[]
     # Mehmet Pekmezci. : make combination 
     for i in range(self.word2vec_window_size):
      row_i=x_data[:,i,:]
      x_data[:,i,:]=x_data[:,int((i+1)%self.word2vec_window_size),:]
      x_data[:,int((i+1)%self.word2vec_window_size),:]=row_i
      x_data_window=np.reshape(x_data,(self.mini_batch_size,int(self.number_of_time_slices/self.word2vec_window_size),self.word2vec_window_size,self.time_slice_length))
      ## switch axes of batch_size and parallel_lstms, then convert it to list according to first axis. --> this will give us list of matrices of shape (mini_batch_size,lstm_time_steps,time_slice_lentgh)
      x_list=np.swapaxes(x_data_window,0,1).tolist()
      result=result+x_list
     return np.random.permutation(result)

 def augment_parallel(self,x_data):
       choice=np.random.rand()*20
       # 10 percent of being not augmented , if equals 0, then not augment, return directly real value
       if choice%10 != 0 :
         SPEED_FACTOR=0.6+choice/20*0.8
         TRANSLATION_FACTOR=int(1000*choice)+1
         ZERO_INDEX=int(choice*1000)+1
         OCCLUDE_START_INDEX=int(choice*3000)+1
         OCCLUDE_WIDTH=2000
         INVERSE_FACTOR=choice%2
         if INVERSE_FACTOR == 1 :
          x_data=-x_data
         x_data=self.augment_speedx(x_data,SPEED_FACTOR)
         x_data=self.augment_translate_and_set_zero_and_occlude(x_data,TRANSLATION_FACTOR,ZERO_INDEX,OCCLUDE_START_INDEX,OCCLUDE_WIDTH)
         #x_data=self.augment_volume(x_data,VOLUME_FACTOR) 
         
 def augment_random(self,x_data):
    augmented_data= np.zeros([x_data.shape[0],x_data.shape[1]],np.float32)
    thread_list=[]
    for i in range(x_data.shape[0]) :
       augmented_data[i]=x_data[i]
       t=threading.Thread(target=self.augment_parallel, args=(augmented_data[i],))
       t.start()
       thread_list.append(t)
    for t in thread_list:
       t.join()   
    return augmented_data

'''
 def generate_single_synthetic_sample(self,single_data):
    generated_data=single_data.copy()
    randomValue=np.random.rand()
    number_of_frequencies=int(randomValue*20)
    #print("generated_data[0:TIME_SLICE]="+str(generated_data[0:TIME_SLICE]))
    #print("number_of_frequencies:"+str(number_of_frequencies))
    for i in range(number_of_frequencies):
      randomValue=np.random.rand()
      frequency=randomValue*10000 # this generates 0-10000 float number,  from uniform dist.
                                  #  frequencies between 10000-20000 is not heard well . so we ignore them. Also sampling rate 22050 only allows to detect TIME_SLICE frequency.
      duration=randomValue*4 # this generates 0-4 float number,  from uniform dist.
      volume=randomValue*5
      #volume=5
      sine_cosine_choice=int(randomValue*2)
      frequency_data=2*np.pi*np.arange(88200)*frequency/22050
      if sine_cosine_choice == 0 :
          wave_data = (np.sin(frequency_data)).astype(np.float32)
      else :
          wave_data = (np.cos(frequency_data)).astype(np.float32)
      current_frequency_data=volume*wave_data
      start_point=int(randomValue*2000)
      #start_point=0
      #if start_point <= self.time_slice_length :
      #   print("frequency-"+str(i)+":"+str(frequency)+"  start_point:"+str(start_point))
      generated_data[start_point:start_point+current_frequency_data.shape[0]]+=current_frequency_data[0:int(current_frequency_data.shape[0]-start_point)]
      #print("generated_data[0:TIME_SLICE]="+str(generated_data[0:TIME_SLICE]))
    return generated_data

 def augment_random(self,x_data):
    augmented_data=np.zeros([x_data.shape[0],x_data.shape[1]],np.float32)
    for i in range(x_data.shape[0]) :
        augmented_data[i]=self.generate_single_synthetic_sample(x_data[i])
    return augmented_data
''' 
