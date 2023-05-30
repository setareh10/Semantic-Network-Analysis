#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:48:35 2020

@author: sr05
"""

import mne
import numpy as np
import os
import sn_config as C
from matplotlib import pyplot as plt
from mne.epochs import equalize_epoch_counts


# path to raw data
data_path = C.data_path
# Parameters
tmin,tmax = C.tmin_cov , C.tmax_cov
subjects = C.subjects
pictures_path = C.pictures_path_grand_average

for i in np.arange(0, len(subjects)):
    print('Participant : ', i)
    meg = subjects[i]
    print(meg)
    
    # epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    # epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    # epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    # epoch_fname_LD    = data_path + meg + 'block_LD_epochs-epo.fif'
    epoch_fname_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
    epoch_fname_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'

    epochs_SD = mne.read_epochs(epoch_fname_SD, preload=True)
    epochs_LD = mne.read_epochs(epoch_fname_LD, preload=True)
    # # Loading epoched data 
    # epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    # epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    # epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)
    # epochs_LD    = mne.read_epochs(epoch_fname_LD   , preload=True)
    # epochs_SD    = mne.concatenate_epochs([epochs_fruit, epochs_odour, 
    #                epochs_milk])
    
    
    # epochs_LD_combined = mne.epochs.combine_event_ids(epochs_LD,['visual',
    #                      'hear','hand','neutral','emotional'], {'words':15})
    # epochs_SD_combined = mne.epochs.combine_event_ids(epochs_SD,['visual',
    #                      'hear','hand','neutral','emotional'], {'words':15})
    
    # equalize_epoch_counts([epochs_SD_combined, epochs_LD_combined])
    evoked_SD_words = epochs_SD['words'].average()
    evoked_LD_words = epochs_LD['words'].average()
    
    C.all_evokeds_sd_words.append(evoked_SD_words)
    C.all_evokeds_ld_words.append(evoked_LD_words)

    C.all_sd_words_nave = C.all_sd_words_nave + evoked_SD_words.nave
    C.all_ld_words_nave = C.all_ld_words_nave + evoked_LD_words.nave

    # ## SD Task     
    # # eeg/mag/grad Grand Average for each individual   
    # for i in np.arange(0,len(C.plot_peaks)):
    #     evoked_SD_words.plot_joint( times='peaks',picks = C.plot_peaks[i], 
    #           title='Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - SD_Words ('+ C.plot_peaks[i]+')', '-Nave: ' + 
    #           str(evoked_SD_words.nave )),ts_args=C.ts_args,
    #           topomap_args=C.topomap_args)
        
    #     plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +
    #           '_SD_Words_'+C.plot_peaks[i])   

    
    # ## LD Task 
    # # eeg/mag/grad Grand Average for each individual   
    # for i in np.arange(0,len(C.plot_peaks)):
    #     evoked_LD_words.plot_joint( times='peaks',picks = C.plot_peaks[i], 
    #           title='Participant : {0}{1}{2}'.format(meg[1:11],
    #           ' - SD_Words ('+ C.plot_peaks[i]+')', '-Nave: ' + 
    #           str(evoked_LD_words.nave )),ts_args=C.ts_args,
    #           topomap_args=C.topomap_args)
        
    #     plt.savefig(pictures_path + 'Participant : ' +meg[1:11] +
    #           '_LD_Words_'+C.plot_peaks[i])  
        


# Grand Average across individuals    
Grand_Average_SD_words = mne.grand_average(C.all_evokeds_sd_words)
Grand_Average_LD_words = mne.grand_average(C.all_evokeds_ld_words)
# SD Task    
#eeg/mag/grad Grand Average for each individual   
for i in np.arange(0,len(C.plot_peaks)):
    Grand_Average_SD_words.plot_joint(times=[0.090,0.135,0.198,0.298], picks = C.plot_peaks[i],
                                      title='Grand Average ('+ C.plot_peaks[i]+')- SD_Words - Nave: '+
          str(C.all_sd_words_nave), topomap_args=C.topomap_args, ts_args=C.ts_args)
    
    plt.savefig(pictures_path + 'Grand_Average_SD_Words_'+C.plot_peaks[i])   


## LD Task 
# eeg/mag/grad Grand Average for each individual   
for i in np.arange(0,len(C.plot_peaks)):
    Grand_Average_LD_words.plot_joint(times=[0.090,0.135,0.198,0.298], picks = C.plot_peaks[i],
                                      title='Grand Average ('+ C.plot_peaks[i]+')- LD_Words - Nave: '+
          str(C.all_ld_words_nave), ts_args=C.ts_args, topomap_args=C.topomap_args)
    
    
    plt.savefig(pictures_path + 'Grand_Average_LD_Words_'+C.plot_peaks[i])   

plt.close('all')