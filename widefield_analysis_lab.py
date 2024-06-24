# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:08:12 2024

@author: sp3660
"""

import os
import pickle
from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import javabridge
import bioformats
import caiman as cm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import os
import glob as glb
from pathlib import PurePath
import numpy as np
import yaml
import pickle
import pandas as pd
import wfield
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import zscore
from tifffile import imread
from scipy.interpolate import interp1d
#step by step widefield calcium retinotopic mapping
from scipy.signal import convolve2d
import cv2
import skimage.io as tf
import concurrent.futures
from sklearn.decomposition import PCA
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
from NeuroAnalysisTools import RetinotopicMapping as rm
import caiman as cm
from caiman.motion_correction import MotionCorrect
import matplotlib.patches as patches
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
def play_movie(array,timeaxis=2,fr=300,play=True):
    temp=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies'

    if timeaxis==2:
        cammov = cm.movie( np.transpose(array, (2, 0, 1)))
    elif timeaxis==0:
        cammov=cm.movie(array)
    if play:
        cammov.play(fr=fr,gain=0.1)
    return cammov

def plot_traces_of_areas(movietoplot,squarexcenter=150,squareycenter=250,squareside=10,squaredistance=50,stimonset=0,stimsweep='undetermined'):
    squarexright=squarexcenter+squaredistance
    squareyright=squareycenter+squaredistance
    squarexleft=squarexcenter-squaredistance
    squareyleft=squareycenter-squaredistance
    squarextop=squarexcenter+squaredistance
    squareytop=squareycenter-squaredistance
    squarexbottom=squarexcenter-squaredistance
    squareybottom=squareycenter+squaredistance
    allsquares=([squarexcenter,squarexright,squarexleft,squarextop,squarexbottom],[squareycenter,squareyright,squareyleft,squareytop,squareybottom])
    

    colors=['blue','tab:orange','gold','tab:red','tab:purple']
    labels=['center','right','left','top','bottom']
   
    f,ax=plt.subplots()
    f.suptitle(stimsweep)
    ax.imshow(movietoplot.mean(axis=0))
    rects=[]
    for i in range(5):
        rect = patches.Rectangle((allsquares[0][i], allsquares[1][i]), squareside, squareside, linewidth=1,edgecolor=colors[i], facecolor='none',label=labels[i])
        rects.append(rect)
        ax.add_patch(rect)
        
    f2,ax2=plt.subplots()
    f2.suptitle(stimsweep)
    for i in range(5):
        ax2.plot(movietoplot[:,allsquares[0][i]:allsquares[0][i]+squareside,allsquares[1][i]:allsquares[1][i]+squareside].mean(axis=(1,2)),color=colors[i],label=labels[i])
        # print(movietoplot[allsquares[0][i]:allsquares[0][i]+squareside,allsquares[1][i]:allsquares[1][i]+squareside,0])
        # print(movietoplot[allsquares[0][i]:allsquares[0][i]+squareside,allsquares[1][i]:allsquares[1][i]+squareside,1])
    ax2.axvline(stimonset)
    ax2.plot(movietoplot.mean(axis=(1,2)),color='k',label='full')

    ax2.legend()
    
    return rects
   
diposlog=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\AWFIELD\QFIELD\20230326\B7M1\visual_display_log\230326183902-KSstimAllDir-MMOUSE-USER-TEST-notTriggered-complete.pkl'
nidaq=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\AWFIELD\NIDAQ\20230326\B7M1_20230326_000.csv'
videofile=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\AWFIELD\QFIELD\20230326\B7M1\B7M1_2023032600000.tif'
#%% visstim
obj = pd.read_pickle(diposlog)

np.diff(obj['presentation']['frame_ts_start'])


seq=pd.DataFrame(obj['presentation']['displayed_frames'])

all_offsets={}
all_onsets={}
for treat in set(seq[4].values): 
    tt=(seq[4]==treat)
    zz=tt.to_numpy()
    onsets=signal.find_peaks(np.diff(zz))[0]
    if treat=='B2U':
        onsets=onsets-onsets[0]
        
    else:
        onsets=onsets+1
    all_onsets[treat]=onsets[::2]
    all_offsets[treat]=onsets[1::2]


order=['B2U','U2B','L2R','R2L']*15

#%% signals
imagingfrequency=10#hz
signals=pd.read_csv(nidaq)
for col in signals.columns:
    print(col)
sig=signals.to_numpy()

time=sig[:,0]
visstim=sig[:,1]
frames=sig[:,2]

frame_peaks=np.diff(frames,prepend=0)
# f,ax=plt.subplots()
# # ax.plot(time[:100000],visstim[:100000]+3)
# ax.plot(time[100000:],frames[100000:])
# ax.plot(time[100000:],frame_peaks[100000:])
onsetframes=signal.find_peaks(frame_peaks,1)[0]

onsets_times=time[onsetframes]
#%% diode signal
filtered=signal.medfilt(visstim,11)
# f,ax=plt.subplots()
# ax.plot(time,visstim)
# ax.plot(time,filtered)


stimonsets=signal.find_peaks(filtered,0.8)[0]
stim_onset_times=time[stimonsets]

preframes=20
stimframes=150


stim_onest_frames=[]
for stimon in stimonsets:
    stim_onest_frames.append(np.where((stimon-onsetframes)>0)[0][-1])

#%% video 

video=cm.load(videofile)


#%% trial slicin
order=['B2U','U2B','L2R','R2L']*15

sliced_dict={'B2U':[],'U2B':[],'L2R':[],'R2L':[]}
for j, i in enumerate(stim_onest_frames):
    sliced_dict[order[j]].append(video[i-preframes:i+stimframes,:,:])

all_trialaveraged={}
all_trialaveraged_dff={}

for treat,trials in sliced_dict.items():
    treatment_trials=np.stack(trials,axis=0)
    trialaveraged=treatment_trials.mean(axis=0)
    all_trialaveraged[treat]=trialaveraged
    all_trialaveraged_dff[treat]=trialaveraged/trialaveraged[:preframes].mean(axis=0)-1
    #%%
    
mov=all_trialaveraged_dff['L2R']
    

# test=play_movie(mov,timeaxis=0,fr=30,play=True)
# rects=plot_traces_of_areas(test,squarexcenter=150,squareycenter=350,squareside=2,squaredistance=20,stimonset=preframes)
#%%
phase_maps={}
powe_maps={}
for k,item in all_trialaveraged_dff.items():
    spectrumMovie = np.fft.fft(item, axis=0)
    #generate power movie
    powerMovie = (np.abs(spectrumMovie) * 2.) / np.size(item, 0)
    powerMap = np.abs(powerMovie[1,:,:])
    freqs = np.fft.fftfreq(item.shape[0])

     
    #generate phase movie
    phaseMovie = np.angle(spectrumMovie)
    phaseMap = -1 * phaseMovie[1,:,:]
    phaseMap = phaseMap % (2 * np.pi)
    
    phase_maps[k]=phaseMap
    powe_maps[k]=powerMap
    



     
f,ax=plt.subplots(2,2)
for i,item in enumerate(phase_maps.values()):
    im=ax.flatten()[i].imshow(item  ,cmap='hsv')

f,ax=plt.subplots(1,2)
im=ax[0].imshow(phase_maps['R2L']-phase_maps['L2R']   ,cmap='hsv',vmin=0, vmax=2*np.pi)
im=ax[1].imshow(phase_maps['B2U']-phase_maps['U2B']   ,cmap='hsv',vmin=0, vmax=2*np.pi)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = f.colorbar(im, cax=cax)
cbar.set_label('Value')

altitude_map=phase_maps['B2U']-phase_maps['U2B']
azimuth_map=phase_maps['R2L']-phase_maps['L2R']  


#%%
k='B2U'
item=all_trialaveraged_dff[k]
spectrumMovie = np.fft.fft(item, axis=0)
freqs = np.fft.fftfreq(item.shape[0])

for p in [225,250,275,300,]:
    central_pixel = (p,p)
    central_signal = item[:, central_pixel[0], central_pixel[1]]
        
        
    sorted_indices = np.argsort(np.abs(spectrumMovie[:, central_pixel[0], central_pixel[1]]))[::-1]
    sorted_freqs = freqs[sorted_indices]
    sorted_amplitudes = np.abs(spectrumMovie[sorted_indices, central_pixel[0], central_pixel[1]])
    sorted_phases = np.angle(spectrumMovie[sorted_indices, central_pixel[0], central_pixel[1]])
    
    
    # Find frequency with the highest amplitude
    max_freq_index = np.argmax(np.abs(spectrumMovie[:len(freqs)//2, central_pixel[0], central_pixel[1]]))
    if max_freq_index==0:
        max_freq_index=1
    max_freq = freqs[max_freq_index]
    max_phase = np.angle(spectrumMovie[max_freq_index, central_pixel[0], central_pixel[1]])
    
    # Reconstruct signal using only the highest power frequency
    reconstructed_signal = np.fft.ifft(np.eye(len(freqs))[:, max_freq_index] * spectrumMovie[:, central_pixel[0], central_pixel[1]])
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot signal of the central pixel
    axs[0].plot(central_signal)
    axs[0].set_title('Signal of Central Pixel')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Intensity')
    
    # Plot frequency decomposition of the central signal in logarithmic scale
    axs[1].plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(spectrumMovie[:len(freqs)//2, central_pixel[0], central_pixel[1]])))
    axs[1].set_title('Frequency Decomposition of Central Pixel Signal (Logarithmic)')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Amplitude (dB)')
    # Plot the original signal overlaid with the reconstructed signal
    axs[2].plot(central_signal, label='Original Signal')
    axs[2].plot(reconstructed_signal.real, label=f'Reconstructed Signal (freq={max_freq}, phase={max_phase:.2f})')
    axs[2].set_title('Original Signal and Reconstructed Signal with Highest Power Frequency')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Intensity')
    max_reconstructed_index = np.argmax(reconstructed_signal.real)
    axs[2].axvline(x=max_reconstructed_index, color='r', linestyle='--', label=f'Max of Reconstructed Signal')
    plt.show()
#%%
exps=['R2L', 'U2B', 'L2R', 'B2U']

plt.close('all')
for stim in range(4):
    
    trialaversgeddff=all_trialaveraged_dff[exps[stim]]
        
    # cammovie=play_movie(trialaversgeddff,timeaxis=0,fr=300)
    test=play_movie(trialaversgeddff,timeaxis=0,fr=30,play=False)
    # smoothed=cm.movie(spatially_smooth_timeseries(trialaversgeddff,axis=0,sigma=1.5))
    # test=play_movie(smoothed,timeaxis=0,fr=30,play=True)

    rects=plot_traces_of_areas(test,squarexcenter=200,squareycenter=300,squareside=2,squaredistance=20,stimsweep=exps[stim],stimonset=preframes)


#%%

phase_ranges=-np.linspace(-np.pi,np.pi,180)


#%%

example_folder = r'C:\Users\sp3660\Documents\Github\NeuroAnalysisTools\NeuroAnalysisTools\test\data'

vasculature_map = tf.imread(os.path.join(example_folder, 'example_vasculature_map.tif'))


params = {
          'phaseMapFilterSigma': 1,
          'signMapFilterSigma': 1,
          'signMapThr': 0.2,
          'eccMapFilterSigma': 15.0,
          'splitLocalMinCutStep': 5.,
          'closeIter': 3,
          'openIter': 3,
          'dilationIter': 15,
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1,
          'mergeOverlapThr': 0.1
          }



trial = rm.RetinotopicMappingTrial(altPosMap=altitude_map,
                                   aziPosMap=azimuth_map,
                                   altPowerMap=None,
                                   aziPowerMap=None,
                                   vasculatureMap=None,
                                   mouseID='test',
                                   dateRecorded='160612',
                                   comments='This is an example.',
                                   params=params)



_ = trial._getSignMap(isPlot=True)
plt.show()