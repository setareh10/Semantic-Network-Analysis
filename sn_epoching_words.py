#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:48:20 2020

@author: sr05
"""
import numpy as np
import mne
import sn_config as C
from mne.epochs import equalize_epoch_counts
from joblib import Parallel, delayed
import time

# path to raw data
data_path = C.data_path


# subjects' directories
subjects =  C.subjects 

def SN_epoching_words(i):
    print('Participant : ', i)
    meg = subjects[i]
    
    
    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    epoch_fname_LD    = data_path + meg + 'block_LD_epochs-epo.fif'
    
    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)
    epochs_LD    = mne.read_epochs(epoch_fname_LD   , preload=True)

    # epochs_SD    = mne.concatenate_epochs([epochs_fruit, epochs_odour, 
    #                epochs_milk])
    
    # epochs_SD_words = mne.epochs.combine_event_ids(epochs_SD,['visual',
    #                      'hear','hand','neutral','emotional'], {'words':15})
    epochs_LD_words = mne.epochs.combine_event_ids(epochs_LD,['visual',
                          'hear','hand','neutral','emotional'], {'words':15})
    
    # epochs_fruit_words = mne.epochs.combine_event_ids(epochs_fruit,['visual',
    #                      'hear','hand','neutral','emotional'], {'words':15})
    # epochs_odour_words = mne.epochs.combine_event_ids(epochs_odour,['visual',
    #                      'hear','hand','neutral','emotional'], {'words':15})    
    # epochs_milk_words  = mne.epochs.combine_event_ids(epochs_milk,['visual',
    #                      'hear','hand','neutral','emotional'], {'words':15})
    
    # equalize_epoch_counts([epochs_SD_words, epochs_LD_words])
    # out_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
    # out_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
    
    # epochs_SD_words.save(out_name_SD, overwrite=True)
    # epochs_LD_words.save(out_name_LD, overwrite=True)
    
    # out_name_fruit = data_path + meg + 'block_fruit_words_epochs-epo.fif'
    # out_name_odour = data_path + meg + 'block_odour_words_epochs-epo.fif'
    # out_name_milk  = data_path + meg + 'block_milk_words_epochs-epo.fif'
    out_name_LD  = data_path + meg + 'block_LD_words_epochs-epo.fif'

    # epochs_fruit_words.save(out_name_fruit, overwrite=True)
    # epochs_odour_words.save(out_name_odour, overwrite=True) 
    # epochs_milk_words.save(out_name_milk  , overwrite=True) 
    epochs_LD_words.save(out_name_LD , overwrite=True) 

s=time.time()
GOF_ave=Parallel(n_jobs=-1)(delayed(SN_epoching_words)(i)for i in np.arange(len(subjects)))    
e=time.time()
print(e-s)

