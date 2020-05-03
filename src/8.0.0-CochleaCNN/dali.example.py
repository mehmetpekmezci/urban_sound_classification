from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
import librosa 
import numpy as np

class MFCCPipeline(Pipeline):
    def __init__(self, device, batch_size, nfft, window_length, window_step,
                 dct_type, n_mfcc, normalize, lifter, num_threads=1, device_id=0):
        super(MFCCPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device

        self.batch_data = []
        y, sr = librosa.load(librosa.util.example_audio_file())
        for _ in range(batch_size):
            self.batch_data.append(np.array(y, dtype=np.float32))

        self.external_source = ops.ExternalSource()
        self.spectrogram = ops.Spectrogram(device=self.device,
                                           nfft=nfft,
                                           window_length=window_length,
                                           window_step=window_step)

        self.mel_fbank = ops.MelFilterBank(device=self.device,
                                           sample_rate=sr,
                                           nfilter = 128,
                                           freq_high = 8000.0)

        self.dB = ops.ToDecibels(device=self.device,
                                 multiplier = 10.0,
                                 cutoff_db = -80.0)

        self.mfcc = ops.MFCC(device=self.device,
                             axis=0,
                             dct_type=dct_type,
                             n_mfcc=n_mfcc,
                             normalize=normalize,
                             lifter=lifter)

    def define_graph(self):
        self.data = self.external_source()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.spectrogram(out)
        out = self.mel_fbank(out)
        out = self.dB(out)
        out = self.mfcc(out)
        return out

    def iter_setup(self):
        self.feed_input(self.data, self.batch_data)
        
        
        
        

n_fft=512
hop_length=512
pipe = MFCCPipeline(device='cpu', batch_size=1, nfft=n_fft, window_length=n_fft, window_step=hop_length,
                    dct_type=2, n_mfcc=40, normalize=True, lifter=0)
pipe.build()
outputs = pipe.run()
mfccs_dali = outputs[0].at(0)
print(mfccs_dali.shape)
