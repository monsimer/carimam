# ------
# Modified by Audrey Monsimer
# September 2021
# ------

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import umap
import hdbscan
import soundfile as sf
import sys
import seaborn as sns
from scipy import signal
import pickle
import pywt
import soundfile
from scipy.io import wavfile
import scipy.io
from spectral_extraction import configs



site=str(sys.argv[1])  #'LOT2/ANGUILLA_20210128_20210318' 
configID=str(sys.argv[2])  #'mel_HBF'
cluster_size=100
samples=30
# cluster_size=int(sys.argv[3])
# samples=int(sys.argv[4])

'''
This file compute the clustering of spectrograms, depend to their configuration
'''

folder ='/bettik/PROJECTS/pr-carimam/monsimau/script/spectral_extraction/results/'+site+'/'  # path to a given recording station folder
outfolder = '/bettik/PROJECTS/pr-carimam/monsimau/script/sort_cluster/results/'+site+'/'  # path to the folder to print clustered pngs
folder_sons='/bettik/PROJECTS/pr-carimam/COMMON/data/'+site+'/'

sort = True # whether we sort each frequency bins by descending order (used to build an energy distribution image and eliminate time dependent features)
# fs = 512000 # TODO adapt to each recording station (some run at 256kHz)
chunksize = 20 # same as in sort_cluster.py, not in seconds
winsize = 1024 if configID != 'stft_HF' else 256 # global STFT window size (we change the sample rate to tune the freq / time resolutions)
hopsize = winsize//2


config = configs[configID.split('_')[1]]
mfs = config['fs']
#get filenames list, filter _spec.npy files only (possibly sample a subset randomly for testing)
fns = pd.Series(os.listdir(folder))
fns = fns[fns.str.endswith('_spec.npy')] #.sample(500)

# # for each configuration, we load spectrograms, project features, cluster, and plot spectrograms in pngs
# for config in configs[1:]:
    # arrays X and meta will hold features and metadata for each samples to be projected / clustered
X, meta = [], []
for f in tqdm(fns, desc='loading spectros for '+configID, leave=False):
    if len(f)>10:
        # load the spectrogram from the .npy file
        spectro = np.load(folder+f, allow_pickle=True).item()[configID]
        fs = np.load(folder+f, allow_pickle=True).item()['fs']
        wavfn = f.rsplit('_', 1)[0] + '.WAV'
        fileDur = sf.info(folder_sons+f.rsplit('_',1)[0]+'.WAV').duration
        source_fs = sf.info(folder_sons+f.rsplit('_',1)[0]+'.WAV').samplerate
        # cut the spectrogram in chunks time wise
        for offset in np.arange(0, spectro.shape[1]-chunksize, chunksize):
            # get logarithmic magnitude
            temp = np.log10(spectro[:,offset:offset+chunksize])
            if sort:
                # if sort is True, we sort each frequency bin in descending order
                temp = np.flip(np.sort(temp, axis=1), axis=1)
                # we can then select a subset of bins (similar to quantiles)
                X.append(temp[:, [1, 3, 5]])
            else :
                # else, we use the whole spectrogram as input features for projection
                X.append(temp)

            
            timeuds = ((fileDur * mfs - winsize) // hopsize +1)//128 # == len(spectro) // 128
            start = offset * timeuds * hopsize / mfs # in seconds, idtimebin * hopsize / fs
            stop = (offset+ chunksize) * timeuds * hopsize / mfs
            # save filename and offset to retrieve the sample later on
            meta.append({'fn':f, 'offset_bins':offset,'offset_start':start,'offset_stop':stop})
            
            
#             '''plot des spectros'''
#             sig, fs = sf.read(folder_sons+wavfn, start=int(start * source_fs), stop=int(stop * source_fs))
#             sig = signal.resample(sig, int((stop-start)*mfs))
#             f, t, spec = signal.stft(sig, fs=mfs, nperseg=winsize, noverlap=winsize//2)
#             spec = np.abs(spec)


# rearange X and meta arrays for easier later use
X = np.array(X).reshape(len(X), -1)
meta = pd.DataFrame().from_dict(meta)

# embed the features using UMAP
project = umap.UMAP()
embed = project.fit_transform(X)
os.system('mkdir -p '+outfolder+configID)
    
#     for cluster_size in [50,75,100,125,150]:
#         for samples in [3,5,10,15,20,30,50]:
#             # cluster the embedings (min_cluster_size and min_samples parameters need to be tuned)
clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=samples).fit(embed) # au depart 50 et 3
filename=outfolder+configID+'/model.sav'
pickle.dump(clusterer,open(filename, 'wb'))
meta['cluster'] = clusterer.labels_
# os.system('mkdir -p '+outfolder+config+'/'+str(cluster_size)+'_'+str(samples))
# display information for the user to check whether the clustering has gone well or not
print('clusters for '+configID)
print(meta.groupby('cluster').agg({'fn':'nunique', 'offset_start':'count'})\
      .rename(columns={'fn':'n unique files', 'offset':'n samples'}))



sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=clusterer.labels_, palette="tab20", s=1)
#         plt.scatter(embed[:,0], embed[:,1], c=clusterer.labels_ , cmap="Paired", s=1)
# plt.colorbar()
plt.show()
plt.savefig(outfolder+configID+'/scatter.png')
plt.close()

meta['coord_x']=embed[:,0]
meta['coord_y']=embed[:,1]
# calcul distance 
D = [np.sqrt( (meta['coord_x'].iloc[i]-np.mean(meta['coord_x'][meta['cluster']==meta['cluster'].iloc[i]]))**2 + (meta['coord_y'].iloc[i]-np.mean(meta['coord_y'][meta['cluster']==meta['cluster'].iloc[i]]))**2) for i in range(len(meta['cluster']))]
meta['D']=D
#calcul rang
clusters=pd.unique(meta['cluster'])
clusters.sort()
rangs=[]
for cluster in clusters:
    rangs.extend(meta[meta['cluster']==cluster].sort_values(by=['D']).reset_index().index.values)
meta=meta.sort_values(by=['cluster','D']).reset_index()
meta['rang']=rangs
meta.drop('index',axis=1,inplace=True)
# meta.set_index('index',inplace=True)
# meta.sort_index(inplace=True)
meta.to_csv(outfolder+configID+'/meta.csv') 

            
            
            
            
            
# For each cluster, we create a folder of png spectrograms
for cluster, grp in meta.groupby('cluster'):

    # if the cluster is more than 300 samples big, we consider it is noise (to tune depending on dataset size)
#     if len(grp) > 500:
#         continue
    # create the cluster folder in the config folder
    os.system('mkdir -p '+outfolder+configID+'/'+str(cluster))
    

    # for each sample in the cluster, we plot the spectrogram for quick visualisation of the clustering result
    for row in tqdm(grp.itertuples(), desc='ploting spectros for cluster '+str(cluster), total=len(grp), leave=False):
        
        
        
        wavfn = row.fn.rsplit('_', 1)[0] + '.WAV'
#         fileDur = sf.info(folder_sons+wavfn).duration
#         print(fileDur)
        source_fs = sf.info(folder_sons+wavfn).samplerate
#         timeuds = ((fileDur * mfs - winsize) // hopsize +1)//128 # == len(spectro) // 128
#         start = row.offset_bins * timeuds * hopsize / mfs # in seconds, idtimebin * hopsize / fs
        start=row.offset_start
        stop=row.offset_stop
#         stop = (row.offset_bins + chunksize) * timeuds * hopsize / mfs
#         print('file : ',wavfn,' start : ',start,' stop : ',stop)
        sig, fs = sf.read(folder_sons+wavfn, start=int(start * source_fs), stop=int(stop * source_fs))
        try:
            sig = signal.resample(sig, int((stop-start)*mfs))

            f, t, spec = signal.stft(sig, fs=mfs, nperseg=winsize, noverlap=winsize//2)
            spec = np.abs(spec)
            if config['mel']:
                spec = np.matmul(config['melbank'], spec)
            plt.imshow(10*np.log10(spec),origin='lower', aspect='auto')
            plt.savefig(outfolder+configID+'/'+str(cluster)+'/'+str('%04d'%row.rang)+'_spectro_'+site+'_'+configID+'_'+row.fn.rsplit('_',1)[0]+'_{:.0f}'.format(start)+'.png')
            plt.close()
            
        except: 
            print('Error with '+row.fn.rsplit('_',1)[0]+'_{:.0f}'.format(start))
        
#         '''version qui tourne'''
#         fs, son = wavfile.read(folder_sons+wavfn)
#         L = 128 # sur 0.5 ms de signal 
#         scales = np.arange(4,36,1) # 32 canaux de scalogramme (support morlet ) 
#         plt.figure()
#         for i in range (1,200001,L):
#             coef, freq = pywt.cwt(son[i:i+L],scales,'morl')
#             plt.subplot(2,1,1)
#             plt.imshow(np.abs(coef))

#             plt.subplot(2,1,2) 	
#             plt.plot(son[i:i+L])
            
#             plt.savefig(outfolder+'morlet_'+site+'_.png')
#             plt.show()
        
        
        
        
# #         if not os.path.isfile(outfolder+config+'/'+str(cluster_size)+'_'+str(samples)+'/'+str(cluster)+'/'+row.fn[:-9]+'_'+str(start)+'.png'):
#         # load the spectrogram from the .npy file
#         spectro = np.load(folder+row.fn, allow_pickle=True).item()[config]
# #                     timeuds = (fileDur * fs // 128 +1)//128 # == len(spectro) // 128
# #                     start = row.offset * timeuds * 128 / fs
#         plt.imshow(10*np.log10(spectro[:, int(row.offset_bins) : int(row.offset_bins+chunksize)]), origin='lower', aspect='auto')

#         # TODO convert the offset (in max pooled time bins) to seconds
#         plt.savefig(outfolder+config+'/'+str(cluster)+'/spectro_'+site+'_'+config+'_'+row.fn.rsplit('_',1)[0]+'_{:.0f}'.format(row.offset_start)+'.png')
#         plt.close()
