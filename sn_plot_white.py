#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:56:55 2020

@author: sr05
"""
import mne
import numpy as np
import os
import sn_config as C
from matplotlib import pyplot as plt


# path to raw data
data_path = C.data_path
# Parameters
tmin,tmax = C.tmin_cov , C.tmax_cov
subjects = C.subjects
pictures_path = C.pictures_path_evoked_white

for i in np.arange(0, len(subjects)):
    print('Participant : ', i)
    meg = subjects[i]
    print(meg)
    
    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    epoch_fname_LD    = data_path + meg + 'block_LD_epochs-epo.fif'
    
    # path to noise covariance matrix
    cov_fname_SD = data_path + meg + 'block_SD-noise-cov.fif'
    cov_fname_LD = data_path + meg + 'block_LD-noise-cov.fif'

   
    # Loading epoched data 
    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)
    epochs_LD    = mne.read_epochs(epoch_fname_LD   , preload=True)
    epochs_SD    = mne.concatenate_epochs([epochs_fruit, epochs_odour, 
                   epochs_milk])
    
    
    epochs_LD_combined = mne.epochs.combine_event_ids(epochs_LD,['visual',
                         'hear','hand','neutral','emotional'], {'words':15})
    epochs_SD_combined = mne.epochs.combine_event_ids(epochs_SD,['visual',
                         'hear','hand','neutral','emotional'], {'words':15})

    evoked_LD_words = epochs_LD_combined['words'].average()
    evoked_SD_words = epochs_SD_combined['words'].average()
    
    # Loading noise covariance matrix
    cov_SD = mne.read_cov(cov_fname_SD)
    cov_LD = mne.read_cov(cov_fname_LD)
    
    fig =  evoked_SD_words.plot_white(noise_cov=cov_SD, show=True)
    fig.suptitle('Participant : ' +meg[1:11]+' - SD Task')
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'evoked_white_SD')   


    fig =  evoked_LD_words.plot_white(noise_cov=cov_SD, show=True)
    fig.suptitle('Participant : ' +meg[1:11]+' - LD Task')
    plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +'evoked_white_LD')   
    plt.close('all')