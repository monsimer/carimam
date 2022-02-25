# -*- coding: utf-8 -*-
"""
@author: loicl et HG LIS CNRS 2021
modified by Audrey M
"""

#%% Importations
import pywt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import next_fast_len
import sys
import os


#%% Parameters
# Choosing a random audio
i=int(sys.argv[1])   # number of first file to read in filetxt
filetxt=str(sys.argv[2])
nb=int(sys.argv[3])    # number of files to read
path_enr="/bettik/PROJECTS/pr-carimam/monsimau/script/scalo_V2_db/"+str(sys.argv[4])

file_list= np.loadtxt(filetxt,dtype='str')

# path_enr="/bettik/PROJECTS/pr-carimam/monsimau/script/scalo_V2_db/results_LOT1_1%_LOT3_2%/"
print(file_list[i:i+nb])
for wav in file_list[i:i+nb]:
    if not os.path.isfile(path_enr+wav.replace('/','-').replace('.WAV','.npy')):
        try :
            signal, sr = librosa.load("/bettik/PROJECTS/pr-carimam/COMMON/data/"+wav, sr=None)

            if sr==512000:
                sr=246000
            #duration = int(sr*2.5)  # 2.5 seconds
            num_scales = 64         # 64 coefs in scaleogram
            order = 5               # order of DB wavelet

            #%% Using pywt

            # To be able to use discret wavelets, replace in pywt/_cwt.py:
            # line 123 : dt_out = dt_cplx if hasattr(wavelet, 'complex_cwt') and wavelet.complex_cwt else dt
            # line 127 :     int_psi = np.conj(int_psi[::way]) if hasattr(wavelet, 'complex_cwt') and wavelet.complex_cwt else int_psi[::way]
                # => Now the function can use discrete wavelets

            # line 37 : def cwt(data, scales, wavelet, sampling_period=1., method='conv', axis=-1, way=1):
                # => Now if way = -1 the function will use a reversed wavelet (on time axis)

            # save and import pywt.
            # Rdy for use.

            scales = np.arange(1,num_scales+1)

            wavelet_type = 'db'+str(order)
            coefs, freqs = pywt.cwt(signal, scales, wavelet_type, sampling_period=1/sr, way=-1)

            # fig, axs = plt.subplots(nrows=2, sharex=True)
            # axs[0].imshow(np.abs(coefs), aspect='auto') 
            # axs[1].plot(signal)
        #     print(coefs.shape)
            np.save(path_enr+wav.replace('/','-').replace('.WAV','.npy'),coefs)
            print('File %s terminé'% (path_enr+wav.replace('/','-').replace('.WAV','.npy')))
        except :
            pass