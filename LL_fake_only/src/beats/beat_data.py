import librosa
import numpy as np
import os   
import soundfile as sf

def get_file_content (self, file_name, data_dir):
        file_open = os.path.abspath(os.path.join(data_dir, file_name + '.flac'))
        wav, org_fs = sf.read(file_open)

        if wav.ndim > 1:
           wav = wav[:,0]
        if org_fs != self.fs:
           wav = librosa.core.resample(wav, org_fs, self.fs)

        wav = np.asarray(wav)
        nTime = np.shape(wav)[0]

        nT = self.set_dur*self.fs
        if nTime < nT:
            while True:
                 wav =  np.concatenate((wav, wav), axis=-1)
                 nTime = np.shape(wav)[0]
                 if nTime > nT:
                     break

        #--- Split into 4-second segment
        nTime = np.shape(wav)[0]
        split_num = 2 + np.floor((nTime-nT)*2/nT) # overlapping
        for m in range(int(split_num)):
            if m == split_num - 1:
                tStop  = nTime
                tStart = nTime - nT
            else:
                tStart = int(m*nT/2)
                tStop  = tStart + nT

            if m == 0:
                mul_seg_sample = wav[tStart:tStop,]
                mul_seg_sample = np.reshape(mul_seg_sample, (1,-1))
            else:
                mul_seg_sample = np.concatenate((mul_seg_sample, wav[tStart:tStop,]), 0)


        #mul_seg_sample = mul_seg_sample.astype(np.float32)
        #mul_seg_sample = torch.from_numpy(mul_seg_sample)
        return mul_seg_sample   #  1x1000