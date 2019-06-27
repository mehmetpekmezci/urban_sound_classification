'''

22050 sampling rate -> 11025 hz en fazla algilayabilir.
x[i]=data
X[i]=energy value for  i*fs/window_size, i=0 to window_size

100Hz : To be able to capture 100Hz wave , we have to read 200 data points at least.
10Hz  : To be able to capture 10Hz wave , we have to read 2000 data points at least.

20-20K Hz human hearing

TIME_SLICE=441 --> optimum value for 22050 ( par raport aux graphs)

'''


import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import librosa as librosa
import pyaudio as pyaudio

TIME_SLICE=440

SOUND_RECORD_SAMPLING_RATE=22050
def play_sound(sound_data):
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SOUND_RECORD_SAMPLING_RATE, output=True)
  stream.write(sound_data[:22050],SOUND_RECORD_SAMPLING_RATE)
  stream.write(sound_data[22050:44100],SOUND_RECORD_SAMPLING_RATE)
  stream.write(sound_data[44100:66150],SOUND_RECORD_SAMPLING_RATE)
  stream.write(sound_data[66150:88200],SOUND_RECORD_SAMPLING_RATE)
  stream.stop_stream()
  stream.close()
  p.terminate()


def generate_single_synthetic_sample():
    generated_data=np.zeros(88200,np.float32)
    randomValue=np.random.rand()
    number_of_frequencies=int(randomValue*50)
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
      if start_point <= TIME_SLICE :
         print("frequency-"+str(i)+":"+str(frequency)+"  start_point:"+str(start_point))
      generated_data[start_point:start_point+current_frequency_data.shape[0]]+=current_frequency_data[0:int(current_frequency_data.shape[0]-start_point)]
      #print("generated_data[0:TIME_SLICE]="+str(generated_data[0:TIME_SLICE]))
    return generated_data

y=generate_single_synthetic_sample()

#play_sound(y)

y=y[0:TIME_SLICE]
#play_sound(y)

#print(y)

Fs = 22050  # sampling rate
#Ts = 1.0/Fs; # sampling interval
#t = np.arange(0,4,Ts) # time vector
#t1 = np.arange(0,int(4*Fs)) # time vector




n = len(y) # length of the signal
k = np.arange(n)

freqs=np.zeros(n)
for i in range(int(n/2)):
  freqs[i]=i*Fs/n
  freqs[-i]=freqs[i]

#T = n/Fs
#frq = k/T # two sides frequency range
#frq = frq[range(int(n/2))] # one side frequency range



Y = np.fft.fft(y) # fft computing and normalization
Yabs = np.abs(np.fft.fft(y)) # fft computing and normalization
#Y = Y[range(int(n/2))]
fig, ax = plt.subplots(4, 1)
#ax[0].plot(t,y)
ax[0].plot(k,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
#ax[1].plot(Y,abs(Y),'r') # plotting the spectrum
#ax[1].set_xlabel('Freq (Hz)')
#ax[1].set_ylabel('|Y(freq)|')
#Y=Y/100
#print(Y)
ax[1].plot(freqs,Yabs,'r') # plotting the spectrum
ax[1].set_xlabel('FreqAbs')
ax[1].set_ylabel('Amplitude')

ax[2].plot(freqs,Y,'r') # plotting the spectrum
ax[2].set_xlabel('Freq')
ax[2].set_ylabel('Amplitude')


freqsPooled=freqs[::10]
YabsPooled=np.zeros(int(Yabs.shape[0]/10))
for i in range(int(freqsPooled.shape[0])):
    print(Yabs[i:i+10])
    YabsPooled[i]=np.max(Yabs[i:i+10])
    print(YabsPooled[i])

print(YabsPooled)


ax[3].plot(freqsPooled,YabsPooled,'r') # plotting the spectrum
ax[3].set_xlabel('Freq')
ax[3].set_ylabel('Amplitude')


plt.show()





