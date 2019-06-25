import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import librosa as librosa
import pyaudio as pyaudio

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
    number_of_frequencies=int(randomValue*10)
    print("number_of_frequencies")
    print(number_of_frequencies)
    for i in range(number_of_frequencies):
      randomValue=np.random.rand()
      frequency=randomValue*10000 # this generates 0-10000 float number,  from uniform dist.
                                  #  frequencies between 10000-20000 is not heard well . so we ignore them. Also sampling rate 22050 only allows to detect 11025 frequency.
      print("frequency-"+str(i)+":"+str(frequency))
      duration=randomValue*4 # this generates 0-4 float number,  from uniform dist.
      volume=randomValue*5
      sine_cosine_choice=int(randomValue*2)
      frequency_data=2*np.pi*np.arange(88200)*frequency/22050
      if sine_cosine_choice == 0 :
          wave_data = (np.sin(frequency_data)).astype(np.float32)
      else :
          wave_data = (np.cos(frequency_data)).astype(np.float32)
      current_frequency_data=volume*wave_data
      start_point=int(randomValue*2000)
      #start_point=generated_data.shape[0]-current_frequency_data.shape[0]
      print("start_point:"+str(start_point))
      start_point=int(randomValue*start_point)
      generated_data[start_point:start_point+current_frequency_data.shape[0]]+=current_frequency_data[0:int(current_frequency_data.shape[0]-start_point)]
    return generated_data

y=generate_single_synthetic_sample()

#play_sound(y)

y=y[0:110]


Fs = 22050  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,4,Ts) # time vector
t1 = np.arange(0,int(4*Fs)) # time vector

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

#Y = np.fft.fft(y) # fft computing and normalization
Y = np.abs(np.fft.fft(y)) # fft computing and normalization
#Y = Y[range(int(n/2))]
fig, ax = plt.subplots(2, 1)
#ax[0].plot(t,y)
ax[0].plot(k,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
#ax[1].plot(Y,abs(Y),'r') # plotting the spectrum
#ax[1].set_xlabel('Freq (Hz)')
#ax[1].set_ylabel('|Y(freq)|')
Y=Y/100
print(Y)
ax[1].plot(k,Y,'r') # plotting the spectrum
ax[1].set_xlabel('Freq')
ax[1].set_ylabel('Amplitude')
plt.show()
