#!/usr/bin/env python3
from header import *

def parse_audio_files():

    global RAW_DATA_DIR,CSV_DATA_DIR,FOLD_DIRS
    sub4SecondSoundFilesCount=0
    for sub_dir in FOLD_DIRS:
      logger.info("Parsing : "+sub_dir)
      csvDataFile=open(CSV_DATA_DIR+"/"+sub_dir+".csv", 'w')
      csvDataWriter = csv.writer(csvDataFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      for file_path in glob.glob(os.path.join(RAW_DATA_DIR, sub_dir, '*.wav')):
         logger.info(file_path)
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
                logger.info ("Exception :")
                logger.info (e)
      csvDataFile.close()       
      logger.info("sub4SecondSoundFilesCount="+str(sub4SecondSoundFilesCount));  


def prepareData():
    logger.info("Starting to prepare the data ...  ")

    global RAW_DATA_DIR,CSV_DATA_DIR,FOLD_DIRS
    logger.info ("prepareData function ...")
    if not  os.path.exists(RAW_DATA_DIR) :
       if not  os.path.exists(MAIN_DATA_DIR+'/../data/0.raw'):
         os.makedirs(MAIN_DATA_DIR+'/../data/0.raw')   
    if not os.path.exists(MAIN_DATA_DIR+"/0.raw/UrbanSound8K"):
      if os.path.exists(MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz"):
         logger.info("Extracting "+MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz")
         tar = tarfile.open(MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz")
         tar.extractall(MAIN_DATA_DIR+'/../data/0.raw')
         tar.close()
         logger.info("Extracted "+MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz")
      else :   
         logger.info("download "+MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz from http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2  using firefox browser or chromium  and re-run this script")
         # logger.info("download "+MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz from https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound8k.html using firefox browser or chromium  and re-run this script")
         exit(1)
#         http = urllib3.PoolManager()
#         chunk_size=100000
#         r = http.request('GET', 'http://serv.cusp.nyu.edu/files/jsalamon/datasets/content_loader.php?id=2', preload_content=False)
#         with open(MAIN_DATA_DIR+"/0.raw/UrbanSound8K.tar.gz", 'wb') as out:
#          while True:
#           data = r.read(chunk_size)
#           if not data:
#             break
#           out.write(data)
#         r.release_conn()

    if not os.path.exists(CSV_DATA_DIR) :
       os.makedirs(CSV_DATA_DIR)  
       parse_audio_files()
    if not os.path.exists(NP_DATA_DIR) :
       os.makedirs(NP_DATA_DIR)  
       save_as_np()
    logger.info("Data is READY  in CSV format. ")
    
def save_as_np():
   logger.info ("save_as_np function started ...")
   fold_data_dictionary=dict()
   MAX_VALUE_FOR_NORMALIZATION=0
   MIN_VALUE_FOR_NORMALIZATION=0

   for fold in FOLD_DIRS:
     fold_data_dictionary[fold]=np.array(np.loadtxt(open(CSV_DATA_DIR+"/"+fold+".csv", "rb"), delimiter=","))
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
        
     np.save(MAIN_DATA_DIR+"/2.np/"+fold+".npy",  fold_data_dictionary[fold]) 
     
   np.save(MAIN_DATA_DIR+"/2.np/minmax.npy",[MIN_VALUE_FOR_NORMALIZATION,MAX_VALUE_FOR_NORMALIZATION]) 
   logger.info ("save_as_np function finished ...")

def normalize(data,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION):
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

def load_all_np_data_back_to_memory(fold_data_dictionary):
   
   logger.info ("load_all_np_data_back_to_memory function started ...")
   for fold in FOLD_DIRS:
       logger.info ("loading from "+MAIN_DATA_DIR+"/2.np/"+fold+".npy  ...")
       fold_data_dictionary[fold]=np.load(MAIN_DATA_DIR+"/2.np/"+fold+".npy")
   minmax=np.load(MAIN_DATA_DIR+"/2.np/minmax.npy")
   MIN_VALUE_FOR_NORMALIZATION=minmax[0]
   MAX_VALUE_FOR_NORMALIZATION=minmax[1]

   logger.info ("load_all_np_data_back_to_memory function finished ...")
   return MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION
   

def normalize_all_data(fold_data_dictionary,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION):
     logger.info ("normalize_all_data function started ...")
     for fold in FOLD_DIRS:
       for i in range(fold_data_dictionary[fold].shape[0]) :
          loadedData=fold_data_dictionary[fold][i]
          loadedDataX=loadedData[:4*SOUND_RECORD_SAMPLING_RATE]
          loadedDataY=loadedData[4*SOUND_RECORD_SAMPLING_RATE]
          normalizedLoadedDataX=normalize(loadedDataX,MAX_VALUE_FOR_NORMALIZATION,MIN_VALUE_FOR_NORMALIZATION)
          fold_data_dictionary[fold][i]=np.append(normalizedLoadedDataX,loadedDataY)
     logger.info ("normalize_all_data function finished ...")
     return fold_data_dictionary

def get_fold_data(fold):
     global fold_data_dictionary
     return np.random.permutation(fold_data_dictionary[fold])


def slice_data(data,NUMBER_OF_TIME_SLICES) :
   return np.reshape(data,[-1,NUMBER_OF_TIME_SLICES,int(data.shape[1]/NUMBER_OF_TIME_SLICES)])
  

def augment_speedx(sound_array, factor):
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

def augment_inverse(sound_array):
    return -sound_array

def augment_volume(sound_array,factor):
    return factor * sound_array

def augment_translate(snd_array, n):
    """ Translates the sound wave by n indices, fill the first n elements of the array with zeros """
    new_array=np.zeros(len(snd_array))
    new_array[n:]=snd_array[:-n]
    return new_array


def augment_random(x_data):
  global LAST_AUGMENTATION_CHOICE;

  augmented_data= np.zeros([x_data.shape[0],x_data.shape[1]],np.float32)
  for i in range(x_data.shape[0]) :
    LAST_AUGMENTATION_CHOICE=(LAST_AUGMENTATION_CHOICE+1)%20
    augmented_data[i]=x_data[i]
    # 10 percent of being not augmented , if equals 0, then not augment, return directly real value
    if LAST_AUGMENTATION_CHOICE%10 != 0 :
      SPEED_FACTOR=0.8+LAST_AUGMENTATION_CHOICE/50
      TRANSLATION_FACTOR=int(5000*LAST_AUGMENTATION_CHOICE/10)
      INVERSE_FACTOR=LAST_AUGMENTATION_CHOICE%2
      if INVERSE_FACTOR == 1 :
       augmented_data[i]=-augmented_data[i]
      augmented_data[i]=augment_speedx(augmented_data[i],SPEED_FACTOR)
      augmented_data[i]=augment_translate(augmented_data[i],TRANSLATION_FACTOR)
      #augmented_data[i]=augment_volume(augmented_data[i],VOLUME_FACTOR)
  
  return augmented_data

@numba.njit(parallel = True)
def slice_and_convert_to_gammatone(x_data,mini_batch_size,number_of_time_slices,input_size,SOUND_RECORD_SAMPLING_RATE,GAMMATONE_NUMBER_OF_FILTERS,GAMMATONE_WINDOW_TIME,GAMMATONE_HOP_TIME):
  print(" slice_and_convert_to_gammatone started ")
  low_freq = DEFAULT_LOW_FREQ = 100
  high_freq = DEFAULT_HIGH_FREQ = 44100/4

  fs=SOUND_RECORD_SAMPLING_RATE
  window_time=GAMMATONE_WINDOW_TIME
  hop_time=GAMMATONE_HOP_TIME
  nfilts=channels=GAMMATONE_NUMBER_OF_FILTERS
  f_min=128

  x_data_reshaped = x_data.reshape((mini_batch_size, number_of_time_slices, int(input_size/number_of_time_slices)))
  x_data_gammatone= np.zeros((mini_batch_size,number_of_time_slices,GAMMATONE_NUMBER_OF_FILTERS),np.float32)
  for miniBatch in numba.prange(mini_batch_size):
   for timeSlice in numba.prange(number_of_time_slices):
    wave   = x_data_reshaped[miniBatch,timeSlice]
    wave   = wave.reshape(wave.shape[0])
    #returnvalue = fftweight.fft_gtgram(wave, fs, window_time, hop_time, channels, f_min)
    width  = 1 # Was a parameter in the MATLAB code
    nfft   = int(2**(np.ceil(np.log2(2 * window_time * fs))))
    nwin   = int(np.sign(window_time * fs) * np.floor(np.abs(window_time * fs) + 0.5))
    nhop   = int(np.sign(window_time * fs) * np.floor(np.abs(hop_time    * fs) + 0.5))
    fmax   = fs/2 
    maxlen = nfft/2 + 1 # nyquist.
    ucirc = np.exp(1j * 2 * np.pi * np.arange(0, nfft/2 + 1)/nfft)
##!!!!! !!!!! !!!!! !!!!!    
    ## MEHMET PEKMEZCI : commented out the below line , if we make a new variable, we cannot use the old name whlie using NUMBA !!!!, so i changed it as ucirc_reshaped .
    ## ucirc = ucirc.reshape((1,ucirc.shape[0]))
    ucirc_reshaped = ucirc.reshape((1,ucirc.shape[0]))
##!!!!! !!!!! !!!!! !!!!!    
    ## ERB_SPACE :
    fraction= np.arange(1, nfilts+1)/nfilts
    ear_q = 9.26449 # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1
    centre_freqs=cf_array = ( -ear_q*min_bw + np.exp( fraction * ( -np.log(high_freq + ear_q*min_bw) + np.log(low_freq + ear_q*min_bw))) * (high_freq + ear_q*min_bw))
    T = 1/fs
    erb = width*((centre_freqs/ear_q)**order + min_bw**order)**(1/order)
    B = 1.019*2*np.pi*erb
    arg = 2*centre_freqs*np.pi*T
    vec = np.exp(2j*arg)
    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2*np.cos(arg)/np.exp(B*T)
    B2 = np.exp(-2*B*T)
    rt_pos = np.sqrt(3 + 2**1.5)
    rt_neg = np.sqrt(3 - 2**1.5)
    common = -T * np.exp(-(B * T))

    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = np.array(common * k11)
    A12 = np.array(common * k12)
    A13 = np.array(common * k13)
    A14 = np.array(common * k14)

    gain_arg = np.exp(1j * arg - B * T)
    
    gain = np.abs( (vec - gain_arg * k11) * (vec - gain_arg * k12) * (vec - gain_arg * k13) * (vec - gain_arg * k14) * (  T * np.exp(B*T) / (-1 / np.exp(B*T) + 1 + vec * (1 - np.exp(B*T))))**4)

    ########
    #######
    #######

    #A11, A12, A13, A14 = A11[..., None], A12[..., None], A13[..., None], A14[..., None]
    A11_reshaped, A12_reshaped, A13_reshaped, A14_reshaped, gain_reshaped = np.array(A11).reshape((A11.shape[0],1)), np.array(A12).reshape((A12.shape[0],1)), np.array(A13).reshape((A13.shape[0],1)), np.array(A14).reshape((A14.shape[0],1)), np.array(gain).reshape((gain.shape[0],1))

    r = np.sqrt(B2)
    theta = 2 * np.pi * cf_array / fs
    pole = (r * np.exp(1j * theta))[..., None]
    GTord = 4
    weights = np.zeros((nfilts, nfft))


    weights[:, 0:ucirc_reshaped.shape[1]] = (
          np.abs(ucirc_reshaped + A11 * fs) * np.abs(ucirc_reshaped + A12 * fs)
        * np.abs(ucirc_reshaped + A13 * fs) * np.abs(ucirc_reshaped + A14 * fs)
        * np.abs(fs * (pole - ucirc_reshaped) * (pole.conj() - ucirc_reshaped)) ** (-GTord)
        / gain_reshaped
    )
    maxlen=int(maxlen)
    weights = weights[:, 0:maxlen]

    gt_weights = weights

    ####
    ###
    ###


##############################################################################################################################################
    """ Substitute for Matlab's specgram, calculates a simple spectrogram.
    :param wave: The signal to analyse
    :param nfft: The FFT length
    :param fs: The sampling rate
    :param nwin: The window length (see :func:`specgram_window`)
    :param nhop: The hop size (must be greater than zero)
    """
    # Based on Dan Ellis' myspecgram.m,v 1.1 2002/08/04
    assert nhop > 0, "Must have a hop size greater than 0"
    s = wave.shape[0]

##############################################################################################################################################

    """
    Window calculation used in specgram replacement function. Hann window of
    width `nwin` centred in an array of width `nfft`.
    """
    halflen = nwin/2
    halff = nfft/2 # midpoint of win
    acthalflen = np.floor(min(halff, halflen))
    halfwin = 0.5 * ( 1 + np.cos(np.pi * np.arange(0, halflen+1)/halflen))
    win = np.zeros((nfft,))
    halff=int(halff)
    acthalflen=int(acthalflen)
    win[halff:halff+acthalflen] = halfwin[0:acthalflen];
    win[halff:halff-acthalflen:-1] = halfwin[0:acthalflen];
##############################################################################################################################################
   
    c = 0
    ncols = 1 + np.floor((s-nfft)/nhop)
    ncols=int(ncols)
    d = np.zeros(((1 + int(nfft/2)), ncols), np.dtype(complex))
    for b in range(0, s-nfft, nhop):
      u = win * wave[b:b+nfft]
      t = np.fft.fft(u)
      d[:,c] = t[0:(1+int(nfft/2))].T
      c = c + 1

#    sgram = fftweight.specgram(wave, nfft, fs, nwin, nhop)
    sgram=d
##############################################################################################################################################



    sgram = gt_weights.dot(np.abs(sgram)) / nfft

    ####

    sgram = sgram.reshape((sgram.shape[0]*sgram.shape[1]))
    x_data_gammatone[miniBatch,timeSlice]=sgram
  print(" slice_and_convert_to_gammatone ended ")
  return x_data_gammatone

    # gammatone/fftweight.py 121. satirdan once
    # maxlen=int(maxlen)
    ## 58. satirdan once
    ##s=int(s)
    ##n=int(n)
    ##h=int(h)
    ## 57. satirdan once
    ## ncols=int(ncols)
    ## 57. satirda int(1+n/2)
    ## 62. satirda int(1+n/2)
    ## 29. satirdan once
    ## halff=int(halff)
    ## acthalflen=int(acthalflen)

