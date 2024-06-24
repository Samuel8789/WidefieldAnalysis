# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:41:35 2024

@author: sp3660
"""
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
from scipy.signal import convolve2d, find_peaks
import matplotlib
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
import sys
import time
import glob
import tkinter as tk
from tkinter import filedialog
import gc

def play_movie(array,timeaxis=2,gain=1,fr=300,play=True):

    if timeaxis==2:
        cammov = cm.movie( np.transpose(array, (2, 0, 1)))
    elif timeaxis==0:
        cammov=cm.movie(array)
    if play:
        cammov.play(fr=fr, gain=gain)
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


def hemocorrection(blue_movie, violet_movie,image_info,metadata,smoothing_factor=5):
    
    
    norm_blue = blue_movie.reshape(-1, blue_movie.shape[-1])
    
    norm_violet = violet_movie.reshape(-1, violet_movie.shape[-1])
    to_smoothing=norm_violet
    kernel: npt.NDArray = np.ones((1, smoothing_factor)) / smoothing_factor
    
    padding_begin: npt.NDArray = to_smoothing[:, 0:smoothing_factor - 2].cumsum(axis=1)
    padding_begin2: npt.NDArray = padding_begin[:, ::2] / np.arange(1, smoothing_factor - 1, 2)
    padding_end: npt.NDArray = to_smoothing[:, -1:-(smoothing_factor - 1):-1].cumsum(axis=1)
    padding_end2: npt.NDArray = padding_end[:, ::-2] / np.arange(smoothing_factor - 2, 0, -2)
    
    smoothed: npt.NDArray = convolve2d(to_smoothing, kernel, 'full')
    
    smoothed = np.concatenate(
    [padding_begin2, smoothed[:, smoothing_factor - 1:-smoothing_factor + 1], padding_end2],
    axis=1)
    
    smoothed_violet=smoothed
    
    # # %% correction
    
    to_lstsq: npt.NDArray[np.float64] = np.stack((norm_blue, smoothed_violet, np.ones(norm_blue.shape)),axis=2)
    
    coeffs = []
    for i in range(to_lstsq.shape[0]):
        coeffs.append(np.linalg.lstsq(
            to_lstsq[i, :, 1:],
            to_lstsq[i, :, 0],
            rcond=None)[0] )
    
    linreg_coeffs = np.array(coeffs).reshape(-1, 2)[:, :, np.newaxis]
    
    linreg_prediction: npt.NDArray[np.float64] = smoothed_violet * linreg_coeffs[:, 0, :] + linreg_coeffs[:, 1, :]
    corrected_data: npt.NDArray[np.float64] = norm_blue - linreg_prediction + np.mean(linreg_prediction)
    
    image_hemocorrected = corrected_data.reshape(
                image_info['blue_shape'][0], image_info['blue_shape'][1], -1)
    # f,ax=plt.subplots()
    # ax.plot(xnew,reconstructed_violet.mean(axis=(0,1)))
    # ax.plot(xnew,reconstructed_blue.mean(axis=(0,1)))
    # ax.plot(xnew,image_hemocorrected.mean(axis=(0,1)))
    
    average: npt.NDArray = image_hemocorrected[:, :, :metadata['analog_aligned']['prestim_frames']].mean(2)
    image_output: npt.NDArray = image_hemocorrected - average[:, :, np.newaxis]
    
    return image_hemocorrected, image_output
     

def mot_correct(raw_recording,plot=False,elastic=False):
    
    downsample_ratio = .8  # motion can be perceived better when downsampling in time
    max_shifts = (20, 20)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides =  (5, 5)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (2, 2)  # overlap between pathes (size of patch strides+overlaps)
    pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
    max_deviation_rigid = 20   # maximum deviation allowed for patch with respect to rigid shifts
    shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

    m_orig = cm.movie( np.transpose(raw_recording, (2, 0, 1)))
    m_orig.resize(1, 1, downsample_ratio)

    if plot:
        m_orig.play(q_max=99.5, fr=30, magnification=2) 
        
    fnames=m_orig.save(temp / 'test.mmap')
    
    mc = MotionCorrect(fnames, dview=None, max_shifts=max_shifts,
                      strides=strides, overlaps=overlaps,
                      max_deviation_rigid=max_deviation_rigid, 
                      shifts_opencv=shifts_opencv, nonneg_movie=True,
                      border_nan=border_nan)
    mc.motion_correct(save_movie=True)
    
    # load motion corrected movie
    m_rig = cm.load(mc.mmap_file)
    mot_corrected=m_rig
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
    
    
    if plot:
        # visualize templates
        plt.figure(figsize = (20,10))
        plt.imshow(mc.total_template_rig, cmap = 'gray');
        
        # inspect moviec
        m_rig.resize(1, 1, downsample_ratio).play(
            q_max=99.5, fr=30, magnification=2, bord_px = 0*bord_px_rig) # press q to exit
        
        # plot rigid shifts
        plt.close()
        plt.figure(figsize = (20,10))
        plt.plot(mc.shifts_rig)
        plt.legend(['x shifts','y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels');
        
    if elastic:    
        # motion correct piecewise rigid
        mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
        mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
    
        mc.motion_correct(save_movie=True, template=mc.total_template_rig)
        m_els = cm.load(mc.fname_tot_els)
        mot_corrected=m_els
        
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                          np.max(np.abs(mc.y_shifts_els)))).astype(int)
        final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
        winsize = 100
        swap_dim = False
        resize_fact_flow = .2    # downsample for computing ROF
        
        tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
            fnames, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

        tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
            mc.fname_tot_rig[0], final_size[0], final_size[1],
            swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

        tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
            mc.fname_tot_els[0], final_size[0], final_size[1],
            swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
        
        # print crispness values
        print('Crispness original: ' + str(int(crispness_orig)))
        print('Crispness rigid: ' + str(int(crispness_rig)))
        print('Crispness elastic: ' + str(int(crispness_els)))
        
        #% plot the results of Residual Optical Flow
        fls = [cm.paths.fname_derived_presuffix(mc.fname_tot_els[0], 'metrics', swapsuffix='npz'),
                cm.paths.fname_derived_presuffix(mc.fname_tot_rig[0], 'metrics', swapsuffix='npz'),
                cm.paths.fname_derived_presuffix(mc.fname[0],         'metrics', swapsuffix='npz'),
              ]

    
        if plot:
            m_els.resize(1, 1, downsample_ratio).play(
                q_max=99.5, fr=30, magnification=2,bord_px = bord_px_rig)
    
            cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov*mc.nonneg_movie,
                            m_rig.resize(1, 1, downsample_ratio), m_els.resize(
                        1, 1, downsample_ratio)], axis=2).play(fr=60, q_max=99.5, magnification=2, bord_px=bord_px_rig)
            
            #%visualize elastic shifts
            plt.close()
            plt.figure(figsize = (20,10))
            plt.subplot(2, 1, 1)
            plt.plot(mc.x_shifts_els)
            plt.ylabel('x shifts (pixels)')
            plt.subplot(2, 1, 2)
            plt.plot(mc.y_shifts_els)
            plt.ylabel('y_shifts (pixels)')
            plt.xlabel('frames')
            
            plt.figure(figsize = (20,10))
            plt.subplot(1,3,1); plt.imshow(m_orig.local_correlations(eight_neighbours=True, swap_dim=False))
            plt.subplot(1,3,2); plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
            plt.subplot(1,3,3); plt.imshow(m_els.local_correlations(eight_neighbours=True, swap_dim=False))
            
            plt.figure(figsize = (20,10))
            plt.subplot(211); plt.plot(correlations_orig); plt.plot(correlations_rig); plt.plot(correlations_els)
            plt.legend(['Original','Rigid','PW-Rigid'])
            plt.subplot(223); plt.scatter(correlations_orig, correlations_rig); plt.xlabel('Original'); 
            plt.ylabel('Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
            axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
            plt.subplot(224); plt.scatter(correlations_rig, correlations_els); plt.xlabel('Rigid'); 
            plt.ylabel('PW-Rigid'); plt.plot([0.3,0.7],[0.3,0.7],'r--')
            axes = plt.gca(); axes.set_xlim([0.3,0.7]); axes.set_ylim([0.3,0.7]); plt.axis('square');
            
            plt.figure(figsize = (20,10))
            for cnt, fl, metr in zip(range(len(fls)), fls, ['pw_rigid','rigid','raw']):
                with np.load(fl) as ld:
                    print(ld.keys())
                    print(fl)
                    print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
                          ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
                    
                    plt.subplot(len(fls), 3, 1 + 3 * cnt)
                    plt.ylabel(metr)
                    print(f"Loading data with base {fl[:-12]}")
                    try:
                        mean_img = np.mean(
                        cm.load(fl[:-12] + '.mmap'), 0)[12:-12, 12:-12]
                    except:
                        try:
                            mean_img = np.mean(
                                cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
                        except:
                            mean_img = np.mean(
                                cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]
                                
                    lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
                    plt.imshow(mean_img, vmin=lq, vmax=hq)
                    plt.title('Mean')
                    plt.subplot(len(fls), 3, 3 * cnt + 2)
                    plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
                    plt.title('Corr image')
                    plt.subplot(len(fls), 3, 3 * cnt + 3)
                    flows = ld['flows']
                    plt.imshow(np.mean(
                    np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
                    plt.colorbar()
                    plt.title('Mean optical flow');  
        

    os.remove(fnames)
    os.remove(mc.mmap_file)
    return mot_corrected



def proces_single_color(input_image,metadata,image_info,channel):

    
    baseline_duration=metadata['analog_aligned']['prestim_frames']
    channel_indices=image_info[channel]
    corrected_mask=np.logical_and(channel_indices,image_info['bad_frames_mask'])
    xnew=metadata['analog_aligned']['frame_time'][:image_info['last_accepted_video_frame']]
    x_v=metadata['analog_aligned']['frame_time'][corrected_mask]
    metadata['interpolated_timestamps']=xnew
    input_image_v = input_image[:,:,corrected_mask]

    f = interp1d(x_v, input_image_v,fill_value="extrapolate")
    ynew= f(xnew)
    baseline_mask = np.arange(ynew.shape[2]) < baseline_duration
    df,dff=calculate_dff(ynew,np.argwhere(baseline_mask)[-1][0]+1,time_axis=2)    
    # print(np.abs(np.abs(dff[:,:,-1].mean())-np.abs(dff[:,:,-2].mean())))
    # print(dff[:,:,-2].mean())
    # print(dff[:,:,-3].mean())
    return ynew,dff,df,metadata

  

def normalize_his_equal(trace):
    # % THIS 
    equalized= []
    for i in range(trace.shape[2]):
        normalized_data = cv2.normalize(trace[:, :, i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
        data_uint8 = normalized_data.astype(np.uint8)
    
        equalized.append(cv2.equalizeHist(data_uint8))
        
    
    equalized = np.stack(equalized, axis=2)
    
    # f,ax=plt.subplots(2,2)
    # ax.flatten()[0].imshow(norm_blue[:, :, i])
    # ax.flatten()[1].imshow(normalized_data)
    # ax.flatten()[2].imshow(data_uint8)
    # ax.flatten()[3].imshow(cv2.equalizeHist(data_uint8))

    return equalized

def denoise_pca(trace,n_components=0.9):
    
    reconstructed= []
    for i in range(trace.shape[2]):
    
        to_pca = trace[:, :, i].reshape(-1, 1)
        pca = PCA(n_components=n_components,svd_solver = 'full')
        trained_pca = pca.fit(to_pca)
        reduced_frame = trained_pca.transform(to_pca)
    
        reconstructed_frame = pca.inverse_transform(reduced_frame)
    
        reconstructed_frame = reconstructed_frame.reshape(trace.shape[0], trace.shape[1])
        reconstructed.append(reconstructed_frame)
    
    reconstructed= np.stack(reconstructed, axis=2)
    # f,ax=plt.subplots(1,3)
    # ax.flatten()[1].imshow(reconstructed_frame)
    # ax.flatten()[2].imshow(reconstructed_frame.astype(dtype=np.uint8))
    
    return reconstructed


def gaussian_smooth(input_array, sigma=1, spatial=True, temporal=True,time_axis=2,radius=1):
    if time_axis!=2:
        input_array=np.transpose(input_array, (1, 2, 0))
        
    smoothed_array = input_array.copy()
    if spatial:
        smoothed_array[..., :-1] = gaussian_filter(smoothed_array[..., :-1], sigma=sigma,radius=radius)
    if temporal:
        smoothed_array[..., -1] = gaussian_filter(smoothed_array[..., -1], sigma=sigma,radius=radius)
    return smoothed_array


    # cammovie=play_movie(smoothed_array,timeaxis=2,fr=50,play=False)
    # rects=plot_traces_of_areas(cammovie,squarexcenter=100,squareycenter=100,squareside=1,squaredistance=25,stimonset=59)

def plot_4exp_trials(stack_time_aligned_f, onset,x=150,y=150,width=10):
    
    m = np.s_[x:x+width, y:y+width,:,:]

    arrs=[]
    times=[]
    for i,j in enumerate(stack_time_aligned_f.keys()):
        arrs.append(stack_time_aligned_f[j][m].mean(axis=(0,1)))
        times.append(np.arange(arrs[i].shape[0]) )
   
    f,ax=plt.subplots(2,2,figsize=(10, 6))
    f2,ax2=plt.subplots(2,2,figsize=(10, 6))

    for i,(arr, time) in enumerate(zip(arrs,times)):
    # Plot every single trial along with mean and shaded error bars for each array
    # Plot every single trial for array 1
        for trial in range(arr.shape[-1]):
            ax.flatten()[i].plot(time, arr[:,trial], alpha=0.1)
        
        mean_arr = np.mean(arr, axis=1)
        std_arr = np.std(arr, axis=1)
        ax.flatten()[i].plot(time, mean_arr, label='Array 1 Mean')
        ax2.flatten()[i].plot(time, mean_arr, label='Array 1 Mean')
        ax2.flatten()[i].fill_between(time, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2)
        ax.flatten()[i].axvline(onset)
        ax2.flatten()[i].axvline(onset)




def calculate_dff(mov, onset,time_axis=2):
    
    if len(mov.shape)>1:
    
        _slice=np.s_[:,:,:onset]
    elif time_axis==0 and len(mov.shape)==1:      
        _slice=np.s_[:onset]

    baseline=mov[_slice].mean(axis=time_axis,keepdims=True)
    df=mov-baseline
    dff=df/baseline
    
    return df, dff

def load_voltage_data(analog_file_name:str, plot=False)-> list:
    """
    This function tahes a dat file generated b y the widfeiled imagers and loads it to mempory, it also removes unnecesary dataline
    It used the wfield package
    If required it plots an iverview of the raw data
    It also Zscore the data to center at 0 and make further analysis easier fro the frame deterction,
    Outputs a list with raw and z score data, and analog recording info
    """
    
    analog_data: npt.NDArray
    analog_info: dict[str, any]
    analog_file=wfield.read_imager_analog(analog_file_name),
    analog_data, analog_info = analog_file[0]
    analog_data[0, :] = False
    analog_data: npt.NDArray[np.int32] = analog_data[:-1, :]

    zscored: npt.NDArray[np.uint16] = zscore(analog_data, axis=1)
    if plot:
        f,ax=plt.subplots()
        ax.imshow(analog_data,aspect='auto')
        f,ax=plt.subplots()
        ax.imshow(zscored,aspect='auto')
        plt.show()
    return [analog_data,zscored,analog_info]

def correct_vis_stim_voltage(analog_data:npt.NDArray, analog_info:dict,experimental_info:dict,plot=False):
    """
    This function takes the raw analog data array and cleans the visual stim onset trigger 
    It seems with dual color the voltage signals is altered but with only blue the signal is ok, test with Kengo
    It uses a strategy by davide to detect positive voltage values and select the sequence that has the longest duration.
    I might have done a median filter and get the first peak highr than a thereshold using scipy find_peaks, but leave as is , it works
    
    
    CAREFUL it return a modified analog data array, so rewrieta analog data variable when using the funciton
    """

    analog_in=analog_data[experimental_info['stimulus_line'], :] 
    stimulus: npt.NDArray[np.int32] = (analog_in != 0).astype(np.int32) 
    diff_stimulus: npt.NDArray[np.int32] = np.diff(stimulus)
    
    start_indices: npt.NDArray[np.int32] = np.where(diff_stimulus == 1)[0] + 1
    end_indices: npt.NDArray[np.int32] = np.where(diff_stimulus == -1)[0] + 1
    
    if stimulus[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if stimulus[-1] == 1:
        end_indices = np.append(end_indices, len(stimulus))
    
    # Combine start and end indices into a single array for vectorized operations
    sequences: npt.NDArray[np.int32] = np.vstack((start_indices, end_indices)).T
    
    # Calculate durations and filter sequences
    durations: npt.NDArray[np.int32] = sequences[:, 1] - sequences[:, 0]
    min_duration: int = int(0.002 * experimental_info['analog_sampling_rate'])
    valid_sequences: npt.NDArray[np.int32] = sequences[durations > min_duration]

    # Find the longest sequence
    longest_sequence: npt.NDArray[np.int32] = valid_sequences[np.argmax(durations[durations > min_duration])]
    
    stimulus_clean: npt.NDArray[np.bool_] = np.zeros(analog_in.shape, dtype=np.bool_)
    stimulus_clean[longest_sequence[0]:longest_sequence[1]] = True
    analog_data[experimental_info['stimulus_line'], :]=stimulus_clean
    analog_info['begin_stimulus_idx'] = np.int32(np.argwhere(analog_data[experimental_info['stimulus_line'], :] == 1)[-1])[0]


    if plot:
        f,ax=plt.subplots()
        ax.plot(stimulus_clean)
        ax.plot(analog_in)
        plt.show()
    
    
    return analog_data,analog_info


def binarize_and_detect_last_frames(analog_data:npt.NDArray, zscored:npt.NDArray, analog_info:dict,experimental_info:dict,plot=False):
    """
    This function binarizes the voltage traces correspodning to the frame timestamps for each color and deteccts the last frame recorded for each color.
    It then updates the analog data array and the analog info dictionary
    """
    ## Firs it replaces raw data by zscored
    analog_data[experimental_info['blue_line'], :] = (zscored[experimental_info['blue_line'], :] >= 0)
    analog_data[experimental_info['violet_line'], :] = (zscored[experimental_info['violet_line'], :] >= 0)
    
    

    
    
    last_frame_changes = np.diff(analog_data[[experimental_info['blue_line'], experimental_info['violet_line']], :].astype(np.int8), axis=1)
    if experimental_info['blue_only']:
        analog_info['begin_last_frame_idx'] = np.int32(np.argwhere(np.abs(last_frame_changes[0,:]) == 1)[-1]).flatten()[0]
    else:
        analog_info['begin_last_frame_idx'] = np.int32(np.argwhere(last_frame_changes == 1)[-1, 1]).flatten() 
    
    
    if plot:
        f,ax=plt.subplots()
        ax.plot(analog_data[experimental_info['stimulus_line'], :]+1)
        ax.plot(analog_data[experimental_info['blue_line'], :])
        ax.plot(analog_data[experimental_info['violet_line'], :]+0.2)
        ax.plot(analog_info['begin_last_frame_idx'],analog_data[experimental_info['blue_line']][analog_info['begin_last_frame_idx']],color='c',marker='^',label='begin_last_frame_idx')

        plt.show()
 
    return analog_data,analog_info
    
def cut_end(last_accepted_video_frame,data_to_stimulus):
    """
    This is an utility function to use whith bklue recording only that cycle trhoufh dark frames at the end of the movie until it reaches  the required instensity therehold
    This is because with blue frames the end frame detection is not as clear
    And cutting at the end of the recording is not problemnatic, as wi wil cut even more later
    """
    last_frame_diff=np.abs(np.abs(data_to_stimulus[last_accepted_video_frame])-np.abs(data_to_stimulus[last_accepted_video_frame-1]))
    if last_frame_diff>0.02:
        last_accepted_video_frame=last_accepted_video_frame-1
        last_accepted_video_frame=cut_end(last_accepted_video_frame,data_to_stimulus)
        return last_accepted_video_frame
    else:
        return last_accepted_video_frame
    
    
def create_frame_masks(image_file_name:str,experiment_info:dict,plot=False):
    
    
    """
    This is the fuction to detect the last illuminated frame n the movie and create the blue and violet masks.
    At the end of the movie after
    """
    image_info: dict[str, any]
    image_data: npt.NDArray = np.transpose(imread(image_file_name, maxworkers=8), (1, 2, 0))
    
    
    data_to_stimulus: npt.NDArray = zscore(image_data.mean(axis=(0, 1)))
    
    data_min: np.int32 = data_to_stimulus.min()
    data_size: int = data_to_stimulus.size
    dark_idx_threshold: int = data_size - 10
     
    th=0.75
    
    dark_idx_all_rec: npt.NDArray = np.where(data_to_stimulus < data_min * th)[0] # detects dark frames , maybe skipped, on whole movie
    dark_idx_end_rec: npt.NDArray = dark_idx_all_rec[dark_idx_all_rec > dark_idx_threshold]# detects dark frames only at the end of the movie, this are the ones for alignment purposes
    
    # here create a slincing mask for the dark frames, first the drak frames a the end and the for the potential dark frame sin betwen the movie
    frame_mask: npt.NDArray[np.bool_] = np.ones(data_size, dtype=np.bool_)
    frame_mask[dark_idx_end_rec]: npt.NDArray = False
    frame_mask[dark_idx_all_rec[dark_idx_all_rec < dark_idx_threshold]]: npt.NDArray = False
    last_accepted_video_frame=dark_idx_end_rec[0]-1
    
    if experiment_info['blue_only']:

        last_accepted_video_frame=cut_end(last_accepted_video_frame,data_to_stimulus)
        frame_mask[last_accepted_video_frame:]= False
        blue_mask=frame_mask
        violet_mask=np.full(blue_mask.shape, False)
        bad_frames_mask=blue_mask
        blue_indices = np.nonzero(blue_mask)[0]
        violet_indices = np.nonzero(violet_mask)[0]
        last_frame_color='blue' 
        first_frame_color='blue'
       
    
    else:
    
        initial_blue_mask: npt.NDArray = (data_to_stimulus > 0.2)
        initial_violet_mask: npt.NDArray = np.logical_and(last_accepted_video_frame > data_min * th, data_to_stimulus < 0)
        
        blue_mask: npt.NDArray = initial_blue_mask.copy()
        violet_mask: npt.NDArray = initial_violet_mask.copy()
        
        rolled_blue_mask: npt.NDArray = np.roll(initial_blue_mask, 1)
        rolled_violet_mask: npt.NDArray = np.roll(initial_violet_mask, 1)
        same_color_consecutive: npt.NDArray = (
                (rolled_blue_mask & initial_blue_mask) |
                (rolled_violet_mask & initial_violet_mask)
        )
        same_color_consecutive[0]=False
        frame_mask &= ~same_color_consecutive
        
        bad_frames_mask=np.roll(frame_mask,-1)
        bad_frames_mask[-1]=False
        
        # blue_indices = np.nonzero(frame_mask & initial_blue_mask)[0]
        # violet_indices = np.nonzero(frame_mask & initial_violet_mask)[0]
        
        # if blue_indices.size > violet_indices.size:
        #     frame_mask[blue_indices[-1]] = False
        #     last_accepted_video_frame=blue_indices[-1]-1
        # elif violet_indices.size > blue_indices.size:
        #     frame_mask[violet_indices[-1]] = False
        #     last_accepted_video_frame=violet_indices[-1]-1
        
        blue_mask &= frame_mask
        violet_mask &= frame_mask
        blue_indices = np.nonzero(blue_mask)[0]
        violet_indices = np.nonzero(violet_mask)[0]
        
        last_frame_color='blue' if np.argmin(np.array([blue_indices[0],violet_indices[0]]) )==0 else 'violet'     
        first_frame_color='blue' if np.argmax(np.array([blue_indices[-1],violet_indices[-1]]) ) ==0 else 'violet'

    image_info: dict[any] = {
        'first_black_frame': dark_idx_end_rec[0],
        'last_accepted_video_frame':last_accepted_video_frame,
        'violet_mask': violet_mask,
        'blue_mask': blue_mask,
        'bad_frames_mask':bad_frames_mask,
        'violet_shape': [image_data.shape[0], image_data.shape[1], np.sum(violet_mask)],
        'blue_shape': [image_data.shape[0], image_data.shape[1], np.sum(blue_mask)],
        'full_shape':[image_data.shape[0], image_data.shape[1], last_accepted_video_frame],
        'last_frame_color':last_frame_color ,
        'first_frame_color':first_frame_color    ,
        'bad_frames':np.argwhere(bad_frames_mask[:last_accepted_video_frame]==0).flatten(),
        'end_black_frames_nr':len(dark_idx_end_rec)
    }
 
    if plot:
        f,ax=plt.subplots()
        ax.plot(data_to_stimulus)
        ax.plot(dark_idx_end_rec,data_to_stimulus[dark_idx_end_rec],'ro',label='dark_endframes')
        ax.plot(last_accepted_video_frame,data_to_stimulus[last_accepted_video_frame]+0.05,'yo',label='last_good_frame')
        ax.plot(np.where(blue_mask)[0][-1],data_to_stimulus[np.where(blue_mask)[0][-1]]+0.05,'bo',label='last_blue_frame')
        ax.plot(np.where(blue_mask)[0],data_to_stimulus[blue_mask],'co',label='all_blue_frame')
        if not experiment_info['blue_only']:
            ax.plot(np.where(violet_mask)[0][-1],data_to_stimulus[np.where(violet_mask)[0][-1]]+0.05,color='tab:purple',marker='o',label='last_violet_frame')
            ax.plot(np.where(violet_mask)[0],data_to_stimulus[violet_mask],'mo',label='all_violet_frame')
            ax.plot(np.where(violet_mask)[0],data_to_stimulus[violet_mask],'mo',label='all_violet_frame')
        f.legend()
        plt.show()
      
   
    return image_data,image_info

def align_timestamps_and_stim_onset(time_file:str,analog_data_full, image_info):
    """
    This functions loads movie metadatae(timestamps), and assigns a voltage timesdtanmpo to each frame to detect wich frame includes the stimulus onset,
    Right now the onset is the absolute value of the difference, 
    """
    new_keys: dict = {
               'frameTimes': 'frame_time',
               'imgSize': 'im_size',
               'preStim': 'prestim_samples',
               'postStim': 'poststim_samples',
               'removedFrames': 'removed_frames'
           }  # Laudetur Python, semper laudatur
    
    # Load metadata from the provided MATLAB file
    metadata_in: dict = loadmat(time_file)
    
    # Remove unnecessary metadata fields
    del metadata_in['__globals__'], metadata_in['__header__'], metadata_in['__version__']
    
    metadata: dict = {}
    for key in metadata_in.keys():
        metadata[new_keys[key]] = metadata_in[key]
    
    # Convert 'prestim_samples' and 'poststim_samples' from arrays to integers for easier usage
    metadata['prestim_samples']: np.int32 = np.int32(metadata['prestim_samples'].flatten()[0])
    metadata['poststim_samples']: np.int32 = np.int32(metadata['poststim_samples'].flatten()[0])
    
    # Unpack 'im_size' into a dictionary for better organization
    im_size: npt.NDArray[np.int_] = metadata['im_size'].flatten()
    metadata['im_size'] = {
        'height': im_size[0],
        'width': im_size[1],
        'frames_samples': im_size[-1]
    }
    
    # Extract frame times and convert them to microseconds
    metadata['frame_time']: npt.NDArray = metadata['frame_time'].flatten() * 86400000  # Conversion from Unix time (no unix, but matlabv datenum serial)
   
    
    # from datetime import datetime, timedelta
    # movieonset = datetime.fromordinal(int(metadata['frame_time'].flatten()[0])) + timedelta(days=metadata['frame_time'].flatten()[0]%1) - timedelta(days = 366)
    # sitmonset=visfiles['flip_info']['datetime'][0].to_pydatetime()
    
    # Check for shape consistency in the imported metadata
    if metadata['prestim_samples'] + metadata['poststim_samples'] == metadata['im_size']['frames_samples']:
        pass
    else:
        raise ValueError(
            "Metadata contains inconsistent values, indicating a failure during recording."
            " Please check the metadata for consistency."
        )
        
        
    begin_last_frame_idx=analog_data_full[2].get('begin_last_frame_idx')
    begin_stimulus_idx=analog_data_full[2].get('begin_stimulus_idx')
    last_nonblack_frame=image_info.get('last_accepted_video_frame')
    
    time_conversion_factor: np.float32 = np.float32(1000 / experimental_info['analog_sampling_rate'])
    
    begin_last_frame_time: np.float32 = begin_last_frame_idx * time_conversion_factor
    begin_stimulus_time: np.float32 = begin_stimulus_idx * time_conversion_factor
    
    metadata_ft: npt.NDArray[np.float32] = metadata['frame_time']
    frame_time: npt.NDArray[np.float32] = begin_last_frame_time+1 + metadata_ft- metadata_ft[last_nonblack_frame] 
    
    
    # this method is more simple ,. but depedning ofn  where the alignment fals it could include frames with minimal stimuls overla
    first_stimon: np.int32 = np.argmin(np.abs(frame_time - begin_stimulus_time))
    
    #here by forceing the difference to be positive will always get the first stim onset frame as one with full stimuls overlap
    differences=frame_time - begin_stimulus_time
    negative_indices = np.where(differences > 0)[0]
    first_stimon = negative_indices[np.argmin(np.abs(differences[negative_indices]))]
    
    stimon_frames: npt.NDArray[np.bool_] = np.ones(frame_time.shape, dtype=np.bool_)
    stimon_frames[:first_stimon] = False
    
    metadata_updated: dict[any] = {
        'original': metadata,
        'analog_aligned': dict(frame_time=frame_time, prestim_frames=first_stimon, stimon_frames=stimon_frames)
    }
    
    metadata: dict[str, dict[str, any]]=metadata_updated
    
    stim_info = metadata['analog_aligned']['stimon_frames']
    stim_info = stimon_frames.astype(np.uint8)
    stim_info = stimon_frames[:image_info['last_accepted_video_frame']]

    return metadata,stim_info

def plot_review_summary_of_alignment(analog_data_full, image_data, image_info, metadata,stim_info,experimental_info, trial_nr,interp_blue=np.empty((0,0,0)),interp_violet=np.empty((0,0,0))):
    data_to_stimulus=zscore(image_data.mean(axis=(0, 1)))
    timestamps_nointerp=metadata['analog_aligned']['frame_time']

    if not interp_blue.any():
        proc='no_interp'
    else:
        timestamps_interp=metadata['interpolated_timestamps']
        proc='interp'
        interp_blue=zscore(interp_blue.mean(axis=(0, 1)))
        interp_violet=zscore(interp_violet.mean(axis=(0, 1)))

    plt.ioff()
    f,ax=plt.subplots(figsize=(6,6))
    ax.plot(analog_data_full[0][experimental_info['stimulus_line'], :]+0.5)
    ax.plot(analog_data_full[0][experimental_info['blue_line'], :],'b')
    ax.plot(analog_data_full[2]['begin_last_frame_idx'],analog_data_full[0][experimental_info['blue_line']][analog_data_full[2]['begin_last_frame_idx']],color='c',marker='^',label='begin_last_frame_idx')
    ax.plot(analog_data_full[2]['begin_stimulus_idx'],analog_data_full[0][experimental_info['blue_line']][analog_data_full[2]['begin_stimulus_idx']],color='c',marker='<',label='begin_stimulus_idx')
    ax.plot(timestamps_nointerp,data_to_stimulus)
    ax.plot(timestamps_nointerp[np.arange(image_info['first_black_frame'],image_info['first_black_frame']+image_info['end_black_frames_nr'])],data_to_stimulus[np.arange(image_info['first_black_frame'],image_info['first_black_frame']+image_info['end_black_frames_nr'])],'ro',label='dark_endframes')
    ax.plot(timestamps_nointerp[image_info.get('last_accepted_video_frame')],data_to_stimulus[image_info.get('last_accepted_video_frame')]+0.05,'yo',label='last_good_frame')
    ax.plot(timestamps_nointerp[np.where(image_info['blue_mask'])[0][-1]],data_to_stimulus[np.where(image_info['blue_mask'])[0][-1]]+0.05,'bo',label='last_blue_frame')
    ax.plot(timestamps_nointerp[np.where(image_info['blue_mask'])[0]],data_to_stimulus[image_info['blue_mask']],'co',label='all_blue_frame')
    ax.plot(analog_data_full[2]['begin_stimulus_idx'],analog_data_full[0][experimental_info['stimulus_line'], analog_data_full[2]['begin_stimulus_idx']]+0.5,color='k',marker='^',label='stim_onset')
    ax.plot(timestamps_nointerp[metadata['analog_aligned']['prestim_frames']],data_to_stimulus[metadata['analog_aligned']['prestim_frames']],color='k',marker='^',label='onset_frame')
    ax.legend(loc=0)
    if interp_blue.any():
        ax.plot(timestamps_interp,interp_blue,'-',label='blue_interp')
        ax.plot(timestamps_interp[metadata['analog_aligned']['prestim_frames']],interp_blue[metadata['analog_aligned']['prestim_frames']],color='k',marker='^',label='onset_frame')
        
        
    if not experimental_info['blue_only']:
        ax.plot(analog_data_full[2]['begin_last_frame_idx'],analog_data_full[0][experimental_info['violet_line']][analog_data_full[2]['begin_last_frame_idx']],color='tab:purple',marker='^',label='begin_last_frame_idx')
        ax.plot(analog_data_full[2]['begin_stimulus_idx'],analog_data_full[0][experimental_info['violet_line']][analog_data_full[2]['begin_stimulus_idx']],color='tab:purple',marker='<',label='begin_stimulus_idx')
        ax.plot(analog_data_full[0][experimental_info['violet_line'], :],'tab:purple')
        ax.plot(timestamps_nointerp[np.where(image_info['violet_mask'])[0][-1]],data_to_stimulus[np.where(image_info['violet_mask'])[0][-1]]+0.05,color='tab:purple',marker='o',label='last_violet_frame')
        ax.plot(timestamps_nointerp[np.where(image_info['violet_mask'])[0]],data_to_stimulus[image_info['violet_mask']],'mo',label='all_violet_frame')
        if interp_violet.any():
            ax.plot(timestamps_interp,interp_violet,'-',label='violte_interp')

        
        
    plt.savefig(figure_saver /  f'trial_{trial_nr}_alignment_{proc}.pdf', dpi=300, bbox_inches='tight')
    # plt.close('all')


#%% PATH MANAGING
global  figure_saver,temp,defaultsfolder,example_folder,processed_data

defaultsfolder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Software\Calcium-Imaging-GUI-main\gui\defaults\retinotopy_mapping_factory.yaml'
figure_saver = PurePath(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TrialAlignments')
temp=PurePath(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies')
example_folder = r'C:\Users\sp3660\Documents\Github\NeuroAnalysisTools\NeuroAnalysisTools\test\data'
processed_data=PurePath(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\DataObjects')

#SLECET DATASET
folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Data\29-Apr-2024_1'
folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Data\240601_RetMapping\01-Jun-2024_1'

#%% metadata loading and file listings based on gui, this should be simplified here


def get_some_data_info(folder):
    
    metadata_file =loadmat(glb.glob(os.path.join(folder, '*.mat'))[0])
    times_files: list[str] = glb.glob(
        os.path.join(
            folder,
            'frameTimes_*.mat'
        )
    )
    image_files: list[str] = glb.glob(
        os.path.join(
            folder,
            '*.tif'
        )
    )
    analog_files: list[str] = glb.glob(
        os.path.join(
            folder,
            'Analog_*.dat'
        )
    )
    
    analog_files.sort(
        key=lambda x: int(PurePath(x).stem.split("_")[-1])
    )
    for time, image, analog in zip(times_files, image_files, analog_files):
        time_name: np.int32 = np.int32(PurePath(time).stem.split('_')[-1])
        image_name: np.int32 = np.int32(PurePath(image).stem.split('_')[-1])
        analog_name: np.int32 = np.int32(PurePath(analog).stem.split('_')[-1])
    
        if time_name == image_name and time_name == analog_name:
            pass  # Correct alignment between files
        else:
            raise NameError(
                "There is a mismatch between frame names, image files, and analog files. "
                "A dialog window with Windows File Explorer has been opened. "
                "Check the files."
            )
    
    filepaths=list( zip(times_files, image_files, analog_files))
    n_trials: int = len(filepaths)
    image_dimensions: str = f"{metadata_file['imgSize'][0, 0]} by {metadata_file['imgSize'][0, 1]}"
    baseline_duration: int = metadata_file['preStim'][0, 0]
    
    
    
    names: list[str] = [PurePath(file).stem.split('_')[-1] for file in image_files]
    names: list[int] = list(map(int, names))
    paths= zip(names, zip(times_files, image_files, analog_files))
    #% metadata loading and file listings
    
    ttl_processed_data=PurePath(folder) / PurePath('ttl_output')
    wfield_processed_data=PurePath(folder) 
    
    
    with open(defaultsfolder, 'r') as stream:
        experiment_parameters = yaml.safe_load(stream)
    
    
    stimulus_line=experiment_parameters['STIMULUS_LINE']['default']
    blue_line=experiment_parameters['BLUE_LINE']['default']
    violet_line=experiment_parameters['VIOLET_LINE']['default']
    camera_sampling_rate=experiment_parameters['CAMERA_SAMPLING_RATE']['default']
    analog_sampling_rate=experiment_parameters['ANALOG_SAMPLING_RATE']['default']
    kernel_smoothing_factor=experiment_parameters['KERNEL_SMOOTHING_FACTOR']['default']
    
    
    with open(glb.glob(os.path.join(ttl_processed_data, '*.pkl'))[0], 'rb') as f:
              settings = pickle.load(f)
    
    #% vis stim loading
    
    
    visfiles={}
    for file in glb.glob(os.path.join(ttl_processed_data, '*.csv')):
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        visfiles[ file_name] =df
        del df, file, file_name
    pre_stimulus_dur= 2
    post_stimulus_dur= 6
    visfiles['flip_info']['datetime']= pd.to_datetime( visfiles['flip_info']['datetime'])
       
       # self.directories: dict[str, str] = self._directories_and_import(wfield_processed_data)
    flip_info=visfiles['flip_info']
    phase_map_smoothing_factor: np.int32 = 5
       
       
    frames_before_stimulus_onset: np.int32 = np.int32(pre_stimulus_dur * camera_sampling_rate / 2)
    frames_per_trial: np.int32 = np.int32((pre_stimulus_dur + post_stimulus_dur) * camera_sampling_rate / 2)
    pre_stimulus_dur: np.int32 = np.int32(pre_stimulus_dur)
    post_stimulus_dur: np.int32 = np.int32(post_stimulus_dur)
    camera_sampling_rate: np.float32 = np.int32(camera_sampling_rate)
    analog_sampling_rate: np.float32 = np.int32(analog_sampling_rate)
    stimulus_line: np.int32 = np.int32(stimulus_line)
    blue_line: np.int32 = np.int32(blue_line)
    violet_line: np.int32 = np.int32(violet_line)
    
    kernel_smoothing_factor: np.int32 = np.int32(kernel_smoothing_factor)
    phase_map_smoothing_factor = np.int32(phase_map_smoothing_factor)
    
    grouped = flip_info.groupby(["trial_start", "direction"])
    
    def calculate_sweep_duration(x):
        # x = x[x['phase'] == 'sweep']
        duration = x['datetime'].iat[-1] - x['datetime'].iat[0]
        return np.float32(1/duration.total_seconds())
    
    frequencies: pd.Series = grouped.apply(calculate_sweep_duration)
    frequencies.droplevel(level=0)
    frequencies = frequencies.groupby('direction').mean().round(3)
    sweep_rate= frequencies.to_dict()
    
    #% do some file dataset visstim alignment
    
    zipped: list[tuple[tuple[str, str, str], tuple[int, pd.DataFrame]]] = list(zip(paths, flip_info.groupby('trial_start')))
    
    wfield_id: int
    stimulus_id: int
    wfield_path: tuple
    stimulus_data: tuple
    
    path_stimuli: list[int, str, tuple, pd.DataFrame] = []
    for (wfield_id, wfield_path), (stimulus_id, stimulus_data) in zipped:
      
        path_stimuli.append((wfield_id, stimulus_data['direction'].iat[0], wfield_path, stimulus_data))

    total_files = len(path_stimuli)
    processed_files = 0

    experimental_info={'stimulus_line':stimulus_line,
                       'blue_line':blue_line,
                       'violet_line':violet_line,
                       'kernel_smoothing_factor':kernel_smoothing_factor ,
                       'analog_sampling_rate':analog_sampling_rate
                       }
    
    return experimental_info, path_stimuli


experimental_info, trial_info=get_some_data_info(folder)
#%% GATHERING PARAMETERS AND SELECTING PROCESING OPTIONS
"""
BECAUSE HERE I HAVE GATEHRE INFOP FORM DAVIDES GUI I PUT ALL OF IT IN A DINGLE DICTIONARY TO PASS TO ALL FUNCTIONS THAT MIGHT NEED IT
"""

blue_only=False
if '01-Jun-2024_1' in folder:
    blue_only=True
do_mot_correct=False
smooth=False
do_hist_equal=False
do_pca_denopising=True

experimental_info.update({'blue_only':blue_only,
                   'do_mot_correct':do_mot_correct,
                   'smooth':smooth,
                   'do_hist_equal':do_hist_equal,
                   'do_pca_denopising':do_pca_denopising,                
                   })

#%%
def single_trial_procesing_review():
     # cammovie=play_movie(raw_dff_blue,timeaxis=2,gain=1.1,fr=50,play=True)
     # rects=plot_traces_of_areas(cammovie,squarexcenter=170,squareycenter=170,squareside=10,squaredistance=25,stimonset=59)
    
       # f,ax=plt.subplots()
     # ax.plot(xnew,ynew_v.mean(axis=(0,1)))
     # ax.plot(xnew,ynew_b.mean(axis=(0,1)))
     
     # ax.plot(x_v,input_image_v.mean(axis=(0,1))-100)
     # ax.plot(x_b,input_image_b.mean(axis=(0,1))-100)
     # ax.plot(metadata['analog_aligned']['frame_time'],mot_corrected.mean(axis=(0,1)))
     
     # f,ax=plt.subplots()
     # ax.plot(xnew,norm_violet.mean(axis=(0,1)))
     # ax.plot(xnew,norm_blue.mean(axis=(0,1)))   
     
     # # cammovie=play_movie(dff_equalized_blue,timeaxis=2,fr=100,play=True)
     # # rects=plot_traces_of_areas(cammovie,squarexcenter=150,squareycenter=250,squareside=10,squaredistance=25)
     
        
    # cammovie=play_movie(reconstructed_blue,timeaxis=2,fr=100,play=True)
    # rects=plot_traces_of_areas(cammovie,squarexcenter=150,squareycenter=250,squareside=10,squaredistance=25)
  
     pass
 #%%  MAIN PROCESS FUNCTION LOOP THROUGH ALL TRIALS GET A DATA OBJECT AND THEN REORGANIZE BY THE FORU TRIAL TIPES
    

i=1
path=trial_info[i]
store_all=True # this is to keep al variables for exploration an debugging



def process_all_trials(trial_info,experimental_info):
    results={'raw_blue':[],'raw_blue_df':[],'raw_blue_dff':[],
             'hemo_only':[],'hemo_only_dff':[],
             'full':[],'full_dff':[],
             'no_eq':[],'no_eq_dff':[],
             'no_pca':[],'no_pca_dff':[],
             'raw_blue_eq_denoised':[],'raw_blue_dff_eq_denoised':[],
             'raw_blue_denoised':[],'raw_blue_dff_denoised':[],
             'raw_blue_eq':[],'raw_blue_dff_eq':[],
             }
    
    store_all=False
    for i, path in enumerate(trial_info):
        print(f'Trial_{i+1}')
    
        time_file, image_file_name, analog_file_name = path[2]
        analog_data_full=load_voltage_data(analog_file_name,plot=False)
        analog_data_full[0],analog_data_full[2]=correct_vis_stim_voltage(analog_data_full[0],analog_data_full[2],experimental_info,plot=False)
        analog_data_full[0],analog_data_full[2]=binarize_and_detect_last_frames(analog_data_full[0],analog_data_full[1],analog_data_full[2],experimental_info,plot=False)
        image_data,image_info=create_frame_masks(image_file_name,experimental_info,plot=False)
        metadata,stim_info=align_timestamps_and_stim_onset(time_file,analog_data_full, image_info)
        # plot_review_summary_of_alignment(analog_data_full, image_data, image_info, metadata,stim_info,experimental_info, i)
       
        
        # full processing includes mot correction, normalization(dff), histogram equalization, PCA denoising , temporal smothing of violet and violet substraction of df/f images (id dff befor true) or of raw
        if experimental_info['do_mot_correct']:
            mot_corrected=mot_correct(image_data)
        else:
            mot_corrected=image_data
    
        if experimental_info['smooth']:
            smoothed=gaussian_smooth(mot_corrected,sigma=1.5, spatial=True, temporal=False)
        else:
            smoothed=mot_corrected
            
        raw_blue,raw_dff_blue,raw_df_blue,metadata=proces_single_color(smoothed,metadata,image_info,'blue_mask')
        if not experimental_info['blue_only']:
            raw_violet,raw_dff_violet,raw_df_violet,metadata=proces_single_color(mot_corrected,metadata,image_info,'violet_mask')
       
        # plot_review_summary_of_alignment(analog_data_full, image_data, image_info, metadata,stim_info,experimental_info, i, raw_blue)
        # cammovie=play_movie(raw_dff_blue,timeaxis=2,fr=300)
        
        if experimental_info['do_hist_equal']:
            equalized_blue=normalize_his_equal(raw_blue)
            dff_equalized_blue=normalize_his_equal(raw_dff_blue)
            if store_all or experimental_info['blue_only']:
                results['raw_blue_eq'].append((path[0], path[1], equalized_blue, path[3], stim_info))
                results['raw_blue_dff_eq'].append((path[0], path[1], dff_equalized_blue, path[3], stim_info))
    
            if experimental_info['do_pca_denopising']:
                reconstructed_eq_blue=denoise_pca(equalized_blue)
                reconstructed_eq_blue_dff=denoise_pca(dff_equalized_blue)
                if store_all or experimental_info['blue_only']:
                    results['raw_blue_eq_denoised'].append((path[0], path[1], reconstructed_eq_blue, path[3], stim_info))
                    results['raw_blue_dff_eq_denoised'].append((path[0], path[1], reconstructed_eq_blue_dff, path[3], stim_info))
    
                
        elif experimental_info['do_pca_denopising']:
            reconstructed_raw_blue=denoise_pca(raw_blue)           
            reconstructed_raw_dff_blue=denoise_pca(raw_dff_blue)            
            if store_all or experimental_info['blue_only']:
                results['raw_blue_denoised'].append((path[0], path[1], reconstructed_raw_blue, path[3], stim_info))
                results['raw_blue_dff_denoised'].append((path[0], path[1], reconstructed_raw_dff_blue, path[3], stim_info))
            
                 
        results['raw_blue_dff'].append((path[0], path[1], raw_dff_blue, path[3], stim_info))
        results['raw_blue_df'].append((path[0], path[1], raw_df_blue, path[3], stim_info))
        results['raw_blue'].append((path[0], path[1], raw_blue, path[3], stim_info))
        
        if not experimental_info['blue_only']:
                 
            if experimental_info['do_hist_equal']:
                equalized_violet=normalize_his_equal(raw_violet)
                dff_equalized_violet=normalize_his_equal(raw_dff_violet)
             
                if experimental_info['do_pca_denopising']:
                    reconstructed_eq_violet=denoise_pca(equalized_violet)
                    reconstructed_eq_violet_dff=denoise_pca(dff_equalized_violet)
                
                    full_hemocorrected=hemocorrection(reconstructed_eq_blue,reconstructed_eq_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                    full_hemocorrected_dff=hemocorrection(reconstructed_eq_blue_dff,reconstructed_eq_violet_dff,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                    results['full'].append((path[0], path[1], full_hemocorrected[0], path[3], stim_info))
                    results['full_dff'].append((path[0], path[1], full_hemocorrected_dff[0], path[3], stim_info))
    
                else:
                    
                    no_pca_hemocorrected=hemocorrection(equalized_blue,equalized_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                    no_pca_hemocorrected_dff=hemocorrection(dff_equalized_blue,dff_equalized_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                    
                    results['no_pca'].append((path[0], path[1], no_pca_hemocorrected[0], path[3], stim_info))
                    results['no_pca_dff'].append((path[0], path[1], no_pca_hemocorrected_dff[0], path[3], stim_info))
    
            elif experimental_info['do_pca_denopising']:
                reconstructed_raw_violet=denoise_pca(raw_violet)
                reconstructed_raw_dff_violet=denoise_pca(raw_dff_violet)
            
                no_eq_hemocorrected=hemocorrection(reconstructed_raw_blue,reconstructed_raw_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                no_eq_hemocorrected_dff=hemocorrection(reconstructed_raw_dff_blue,reconstructed_raw_dff_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                
                results['no_eq'].append((path[0], path[1], no_eq_hemocorrected[0], path[3], stim_info))
                results['no_eq_dff'].append((path[0], path[1], no_eq_hemocorrected_dff[0], path[3], stim_info))
                
            else:
                hemocorrect_only=hemocorrection(raw_blue,raw_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
                hemocorrect_only_dff=hemocorrection(raw_dff_blue,raw_dff_violet,image_info,metadata,experimental_info['kernel_smoothing_factor'])
        
                results['hemo_only'].append((path[0], path[1], hemocorrect_only[0], path[3], stim_info))
                results['hemo_only_dff'].append((path[0], path[1], hemocorrect_only_dff[0], path[3], stim_info))
    
        if i==0:
            results = {k: v for k, v in results.items() if v}
        
    return results
    
results=process_all_trials(trial_info,experimental_info)
#%% GENERAL PLAYING AND PLOTTING 

reconstructed_raw_blue

cammovie=play_movie(reconstructed_raw_dff_blue,timeaxis=2,fr=300)
test=play_movie(trialaversgeddff,timeaxis=0,fr=30,play=False)
# smoothed=cm.movie(spatially_smooth_timeseries(trialaversgeddff,axis=0,sigma=1.5))
# test=play_movie(smoothed,timeaxis=0,fr=30,play=True)

rects=plot_traces_of_areas(test,squarexcenter=175,squareycenter=175,squareside=2,squaredistance=20,stimsweep=k,stimonset=onset)

cammovie=play_movie(trace,timeaxis=2,fr=300)

#%% TRY TO SAVRE ALL RESULTS INDIVIDUALLY AS GROUPED STACKS
def save_all_result(results):
    
    def save_results(folder,results,info=''):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        datapath=processed_data /  f'{PurePath(folder).stem}_analysis_{info}_{timestr}.pkl'
        if not os.path.isfile(datapath):
            with open(datapath, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
            
        return datapath
    
    for data_in,stack_in in results.items():

        # data_in='raw_blue_dff'
        # stack_in=results[data_in]
        # del results
        
        keys = set([x[1] for x in stack_in])
        grouped_by_direction = {}
        for key in keys:
            grouped_by_direction[key] = [(x[2], x[3], x[4]) for x in stack_in if x[1] == key]
        stack=grouped_by_direction
        
        datapath=save_results(folder,stack,f'{data_in}_grouped')
        
    del results
    gc.collect()

save_all_result(results)
#%% SAVE A SINGLE SELECTED PROCESSED GROUP STACK

def save_results(folder,results,info=''):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    datapath=processed_data /  f'{PurePath(folder).stem}_analysis_{info}_{timestr}.pkl'
    if not os.path.isfile(datapath):
        with open(datapath, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        
    return datapath


data_in='raw_blue_dff'
stack_in=results[data_in]
del results
gc.collect()

keys = set([x[1] for x in stack_in])
grouped_by_direction = {}
for key in keys:
    grouped_by_direction[key] = [(x[2], x[3], x[4]) for x in stack_in if x[1] == key]
stack=grouped_by_direction
datapath=save_results(folder,stack,f'{data_in}_grouped')


#%% LOAD DATA FROM FILE
def load_data(folder):

    search_string=PurePath(folder).stem
    # Search for pickle files containing the search string
    search_pattern = os.path.join(processed_data, f"*{search_string}*.pkl")
    matching_files = glob.glob(search_pattern)
    
    # Sort files by creation time (most recent first)
    matching_files.sort(key=os.path.getctime, reverse=True)
    
    if not matching_files:
        raise FileNotFoundError(f"No pickle files found matching '{search_string}' in processed_data '{processed_data}'")
    
    # Prompt user to select a file from the sorted list
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Display file selection dialog
    file_path = filedialog.askopenfilename(
        initialdir=processed_data,
        title=f"Select a pickle file containing '{search_string}'",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    
    if not file_path:
        raise FileNotFoundError("No file selected or dialog canceled by user")
    
    # Load the selected pickle file
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    return loaded_data

stack=load_data(folder)

#%% 
def align_and_trial_average(stack):

    def align_trials_onset_and_cut(stimon_info_t, trial_array, all_alignment_info,stimulus):
        
        onsets=[]
        preframe_length_min = np.min(list(map(lambda x: np.sum(x), np.invert(stimon_info_t))))
        onsets.append(preframe_length_min)
        starts=list(map(lambda x: np.sum(x)-preframe_length_min, np.invert(stimon_info_t)))

        onset_aligned=[trial[:,:,starts[i]:] for i,trial in enumerate(trial_array)]

        min_length: np.int32 = np.min([trial.shape[2] for trial in onset_aligned])
        max_length: np.int32 = np.max([trial.shape[2] for trial in onset_aligned])
        
        if stimulus=='left2right' and 'right2left' in all_alignment_info.keys():
             if min_length>all_alignment_info['right2left']['shortest_trial_frames']:
                 min_length=all_alignment_info['right2left']['shortest_trial_frames']
        
        elif stimulus=='right2left' and 'left2right' in all_alignment_info.keys():
            if min_length>all_alignment_info['left2right']['shortest_trial_frames']:
                min_length=all_alignment_info['left2right']['shortest_trial_frames']
            
        elif stimulus=='top2bottom' and 'bottom2top' in all_alignment_info.keys():
            if min_length>all_alignment_info['bottom2top']['shortest_trial_frames']:
                min_length=all_alignment_info['bottom2top']['shortest_trial_frames']
        elif stimulus=='bottom2top' and 'top2bottom' in all_alignment_info.keys():
            if min_length>all_alignment_info['top2bottom']['shortest_trial_frames']:
                min_length=all_alignment_info['top2bottom']['shortest_trial_frames']


        
        stack_cut: list[npt.NDArray[np.float64]] = [trial[:, :, :min_length] for trial in onset_aligned]
        stack_time_aligned: npt.NDArray[np.float64] = np.stack(stack_cut, axis=3)
        
        alignment_info={'all_prestim_frames':list(map(lambda x: np.sum(x), np.invert(stimon_info_t))),
                        'earliest_onset':preframe_length_min,
                        'cut_starts':starts,
                        'shortest_trial_frames':min_length,
                        'longest_trial_frames':max_length                           
                        }
        

        return stack_time_aligned, alignment_info
        
    stack_time_aligned_all={}
    trial_averaged_movies={}
    all_alignment_info={}

    for key, item in stack.items():
        stack_new=list(map(lambda x: x[0], item))
        stimon_info_t = list(map(lambda x: x[2], item))
        
        stack_time_aligned, alignment_info=align_trials_onset_and_cut(stimon_info_t,stack_new,all_alignment_info,key)
        trial_averaged=stack_time_aligned.mean(axis=3)
        
        stack_time_aligned_all[key]=stack_time_aligned
        trial_averaged_movies[key]=cm.movie(np.transpose(trial_averaged, (2, 0, 1)))
        all_alignment_info[key]=alignment_info
        
    return stack_time_aligned_all,trial_averaged_movies, all_alignment_info

stack_time_aligned_all,trial_averaged_movies, all_alignment_info=align_and_trial_average(stack)

#%% soem plotting reviewing trial averaged movies
stimulus='right2left'
stimonset=58
onset=all_alignment_info[stimulus]['earliest_onset']

plt.close('all')
plot_4exp_trials(stack_time_aligned_all,stimonset,x=200,y=200,width=1)
for k,v in trial_averaged_movies.items():
    trialaversgeddff=v
    onset=all_alignment_info[k]['earliest_onset']

    cammovie=play_movie(trialaversgeddff,timeaxis=0,fr=300)
    test=play_movie(trialaversgeddff,timeaxis=0,fr=30,play=False)
    # smoothed=cm.movie(spatially_smooth_timeseries(trialaversgeddff,axis=0,sigma=1.5))
    # test=play_movie(smoothed,timeaxis=0,fr=30,play=True)

    rects=plot_traces_of_areas(test,squarexcenter=175,squareycenter=175,squareside=2,squaredistance=20,stimsweep=k,stimonset=onset)
    # cammovie=play_movie(trialaversgeddff,timeaxis=0,fr=10,play=True)



#%% GET PHASE MAPS BASAED OIN THE FIRST FREQUENCY COMPONENT OF THE FFT, 
 
def get_phase_maps1(movies:dict):
    """
    IThis is what the elife paper suggest, just get the first frequency component, other than 0, is it w=ill be the closest to the stimulus frequency, which is 1/stim time
    """
    
    phase_maps={}
    powe_maps={}
    for k,v in movies.items():
        spectrumMovie = np.fft.fft(v, axis=0)
        freqs = np.fft.fftfreq(v.shape[0])
        
        #generate power movie
        powerMovie = (np.abs(spectrumMovie) * 2.) / np.size(v, 0)
        powerMap = np.abs(powerMovie[1,:,:])
         
        #generate phase movie
        phaseMovie = np.angle(spectrumMovie)
        phaseMap = -1 * phaseMovie[1,:,:]
        phaseMap = phaseMap % (2 * np.pi)
        
        phase_maps[k]=phaseMap
        powe_maps[k]=powerMap
        
    return phase_maps,powe_maps

phase_maps,powe_maps=get_phase_maps1(trial_averaged_movies)
#%% TO DO
def convert_phase_to_screen_pos(phase_maps):
    pass
    

#%% SIGN MAP

altitude_map = phase_maps["top2bottom"] - phase_maps["bottom2top"]
azimuth_map = phase_maps["left2right"] - phase_maps["right2left"]

params = {
          'phaseMapFilterSigma': 3,
          'signMapFilterSigma': 0.5,
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

#%% single trials
k='left2right'
trial=0
recording=stack_time_aligned_all[k][:,:,:,trial].squeeze()
meanimage=recording.mean(axis=2)
stimtable=stack[k][trial][1]
onset=all_alignment_info[k]['earliest_onset']

singletrialmov=play_movie(recording,gain=2,timeaxis=2,fr=300,play=True)
rects=plot_traces_of_areas(singletrialmov,squarexcenter=75,squareycenter=250,squareside=10,squaredistance=25,stimsweep=k,stimonset=onset)

    
#%% PLOTTING THE FFT ANALYSIS WITH A IT MORE DETAIL
k='top2bottom'
item=trial_averaged_movies[k]
spectrumMovie = np.fft.fft(item, axis=0)
freqs = np.fft.fftfreq(item.shape[0])

for p in [150,175,200,225]:
    central_pixel = (p,p)
    central_signal = item[:, central_pixel[0], central_pixel[1]]
        
        
    sorted_indices = np.argsort(np.abs(spectrumMovie[:, central_pixel[0], central_pixel[1]]))[::-1]
    sorted_freqs = freqs[sorted_indices]
    sorted_amplitudes = np.abs(spectrumMovie[sorted_indices, central_pixel[0], central_pixel[1]])
    sorted_phases = np.angle(spectrumMovie[sorted_indices, central_pixel[0], central_pixel[1]])
    
    
    # Find frequency with the highest amplitude
    max_freq_index = np.argmax(np.abs(spectrumMovie[:len(freqs)//2, central_pixel[0], central_pixel[1]]))+1
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
#%% PLOTTING MAPS INDEPENDENTLY
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
f,ax=plt.subplots(2,2)
for i,item in enumerate(phase_maps.values()):
    im=ax.flatten()[i].imshow(item  ,cmap='hsv')
altitude_map=phase_maps['bottom2top']-phase_maps['top2bottom']
azimuth_map=phase_maps['right2left']-phase_maps['left2right'] 
f,ax=plt.subplots(1,2)
im=ax[0].imshow(altitude_map  ,cmap='hsv',vmin=0, vmax=2*np.pi)
im=ax[1].imshow(azimuth_map  ,cmap='hsv',vmin=0, vmax=2*np.pi)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = f.colorbar(im, cax=cax)
cbar.set_label('Value')

 


#%% WAVELET ANALYSIS

import numpy as np
import matplotlib.pyplot as plt
import pywt

# Assuming you have your signal stored in a variable called 'signal'
# For demonstration, let's generate a sample signal
# Replace this with your actual signal data
# signal = ... 
k='right2left'
# Sample signal generation (Replace this with your actual signal)
signal=trial_averaged_movies[k]
#
# smoothed=gaussian_smooth(signal, sigma=0, spatial=False, temporal=True,time_axis=0,radius=0)
smoothed=signal
signal=smoothed[:,170,170]

t = np.linspace(0, 1, len(signal), endpoint=False)

# Perform wavelet transform
wavelet = 'morl'  # Choose a wavelet, here 'morl' is used as an example
scales = np.arange(1, 128)

# f = pywt.scale2frequency(wavelet, scales)/(1/30)
coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot original signal
ax1.plot(t, signal, 'b')
ax1.set_ylabel('Signal')
ax1.set_title('Signal and Wavelet Amplitude-Time Plot')

# Plot amplitude-time plot of the wavelet (scalogram)
extent = [t[0], t[-1], frequencies[-1], frequencies[0]]
img = ax2.imshow(np.abs(coefficients), extent=extent, cmap='jet', aspect='auto')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time')
ax2.set_xlim(t[0], t[-1])


# Add colorbar above the bottom subplot
cbar_bottom = fig.colorbar(img, ax=ax2, orientation='horizontal', pad=0.05)
cbar_bottom.set_label('Magnitude')

plt.show()

