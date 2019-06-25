import pyaudio
import numpy as np
p = pyaudio.PyAudio()
volume = 1     # range [0.0, 1.0]
fs = 22050       # sampling rate, Hz, must be integer
duration = 4.0   # in seconds, may be float
f = 1230.0        # sine frequency, Hz, may be float
# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
# play. May repeat with different volume values (if done interactively) 
stream.write(volume*samples)
stream.stop_stream()
stream.close()
p.terminate()


### NOTE:  3500 Inner Hair Cell, each connected to ~10 neurons, they connect to auditory nucleus, then signals are transferred to the auditory cortex1 then to cortex2
###        Humans can hear 20Hz to 20 000Hz
###        Human  voice frq : 100 to 10000 Hz
###        Human  talk voice frq : 100 to 8000 Hz
###        Asagidaki calismadi:
###        Softmax'de history tut. Weighted Correction by history. Eg. onehot 3'te 0.2 var ama 0 olması lazım, duzeltme olarak 0.2 degil de son K tane errorun ortalamasini verelim. (K=10 ?)

