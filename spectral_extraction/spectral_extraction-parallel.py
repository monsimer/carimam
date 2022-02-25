#-----------
# modified by Audrey Monsimer 
# September 2021
# ----------

import numpy as np
from tqdm import tqdm
import os
from scipy import signal
from scipy.ndimage.filters import maximum_filter
import soundfile as sf
import pandas as pd
import sys

"""
This file computes several spectrograms with a given set of paramters.
Spectrograms of size 128x128 will be saved in .npy files for each found sounfiles found in a given folder
"""

folder = str(sys.argv[1])  # path to a given recording station folder
#'../COMMON/data/BONAIRE/session1_20201217to20210126/' 
fn = str(sys.argv[2])     # filename 
folder_output=str(sys.argv[3])  # path to save the spectrogram 
winsize = 1024 # global STFT window size (we change the sample rate to tune the freq / time resolutions)
# source_fs = 256000 # TODO adapt to each session (some run at 256kHz)

try:
    sig, source_fs = sf.read(folder)
    def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq):
        """
        Creates a mel filterbank of `num_bands` triangular filters, with the first
        filter starting at `min_freq` and the last one stopping at `max_freq`.
        Returns the filterbank as a matrix suitable for a dot product against
        magnitude spectra created from samples at a sample rate of `sample_rate`
        with a window length of `frame_len` samples. 
        """
        # array of triangular filters' peaks (linear scale)
        peaks_mel = np.linspace(1127*np.log1p(min_freq/700), 1127*np.log1p(max_freq/700), num_bands+2)
        # array of triangular filters' peaks (logarithmic scale)
        peaks_hz = 700 * (np.expm1(peaks_mel / 1127))
        peaks_bin = peaks_hz * frame_len / sample_rate

        # spectrogram freq bins indices
        x = np.arange(frame_len//2 + 1).reshape(-1, 1)

        # build triangular filters from left and right boundaries
        l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
        tri_left = (x - l) / (c - l)
        tri_right = (x - r) / (c - r)
        tri = np.minimum(tri_left, tri_right)

        # clip at zero, and normalize each filter by its area
        filterbank = np.maximum(tri, 0)
        filterbank /= filterbank.sum(axis=0)

        # return a weight transformation matrix of shape (num_bands, frame_len//2 + 1)
        return filterbank.T

    """
    Set of spectrogram configurations (sampling rate, wether we yield a regular or mel spectrogram, mel start frequency), 
    each tuned to focus on a given type of vocalisations. e.g. BBF configuration samples at 2kHz. 
    The output mel spectrogram will thus range from 0 to 1kHz, with 512 logarithmic spaced freq bins.
    """
    configs = [
    {'id':'BBF', 'fs': 2000, 'mel':True, 'nomel':False, 'melstart':0}, # tuned for fin whale vocs (~60Hz stationnary)
    {'id':'HBF', 'fs': 16000, 'mel':True, 'nomel':False, 'melstart':0}, # tuned for humpback whale vocs (~500Hz stationnary)
    {'id':'BMF', 'fs': 64000, 'mel':True, 'nomel':False, 'melstart':2000}, # tuned for clicks (sperm whales, delfinids) (~5-30kHz transitory)
    {'id':'HMF', 'fs': 256000 , 'mel':True, 'nomel':True, 'melstart':8000}, # tuned for HF clicks (deflinids, ziphius)
    {'id':'HF', 'fs': 256000, 'mel':False, 'nomel':True} # tuned for very HF clicks (/!\ for this config, we use a 256pts STFT window)
    ]
    # build a low pass filter used before resampling, and the melbank if needed
    for c in configs:
        if c['mel']:
            c['melbank'] = create_mel_filterbank(c['fs'], winsize, 128, c['melstart'], c['fs']//2)
        if c['fs'] < source_fs:
            c['sos'] = signal.butter(3, c['fs']/source_fs, 'lp', output='sos')


    print('doing ', folder)
    
'''
If you want to turn on many files into a folder, you can uncomment these few lines
(and comment line 22 "fn = str(sys.argv[2])")
'''
#     #get filenames list, filter wav files only (possibly sample a subset randomly for testing)
#     fns = pd.Series(os.listdir(folder))
#     fns = fns[fns.str.endswith('WAV')] #.sample(500)

    # for each sound file
    # for fn in tqdm(fns):


    # we build a dictionnary containing a spectrogram for each configuration
    out = {}
    out['fs']=source_fs
    for c in configs:

        # low pass filter at next nyquist frequency and undersample the signal
        if c['fs'] < source_fs:
            csig = signal.sosfiltfilt(c['sos'], sig)
            csig = csig[::(source_fs//c['fs'])]

        # compute the magnitude spectrogram using the STFT
        if c['id'] != 'HF':
            f, t, spec = signal.stft(csig, fs=c['fs'], nperseg=winsize, noverlap=winsize//2)
        else: # special winsize for HF
            f, t, spec = signal.stft(csig, fs=c['fs'], nperseg=256, noverlap=128)
        spec = np.abs(spec)
        spec = 10*np.log10(spec.norm(p=2, axis=-1))
        spec = spec - np.median(spec, axis=1, keepdims=True)[0]
        # we undersample the spectrogram over the time dimension to get 128 time bins only
        time_uds = spec.shape[1]//128
        if c['nomel']:
            out['stft_'+c['id']] = maximum_filter(spec, (1, time_uds))[:128, ::time_uds]
        if c['mel']:
            out['mel_'+c['id']] = maximum_filter(np.matmul(c['melbank'], spec), (1, time_uds))[:,::time_uds]

    # save the dictionnary of spectrograms with at the soundfile location, with the input filename + '_spec.npy'
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    np.save(folder_output+fn.rsplit('.', 1)[0]+'_spec.npy', out)

except:
    print('failed with '+folder)
#     continue
