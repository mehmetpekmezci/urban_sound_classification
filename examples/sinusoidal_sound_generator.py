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
