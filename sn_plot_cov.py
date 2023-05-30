#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:15:13 2020

@author: sr05
"""
import mne
import numpy as np
import os.path as op
import os
import sn_config as C
from mne import compute_covariance


# path to raw data
data_path = C.data_path
# Parameters
tmin,tmax = C.tmin_cov , C.tmax_cov
pictures_path = C.pictures_path_cove
subjects = C.subjects

for i in np.arange(0, len(subjects)):
    print('Participant : ' , i)
    meg = subjects[i]
    
  # Complete path to epoched data 
    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    epoch_fname_LD    = data_path + meg + 'block_LD_epochs-epo.fif'

    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)
    epochs_LD    = mne.read_epochs(epoch_fname_LD   , preload=True)
    epochs_SD    = mne.concatenate_epochs([epochs_fruit, epochs_odour, 
                   epochs_milk])
    
       # path to noise covariance matrix
    cov_fname_SD = data_path + meg + 'block_SD-noise-cov.fif'
    cov_fname_LD = data_path + meg + 'block_LD-noise-cov.fif'

    # Loading noise covariance matrix
    cov_SD = mne.read_cov(cov_fname_SD)
    cov_LD = mne.read_cov(cov_fname_LD)


    fig_cov_SD_fname  = pictures_path + 'Participant : ' +meg[1:11] +'SD_noise_cov'
    fig_spec_SD_fname = pictures_path + 'Participant : ' +meg[1:11] +'SD_noise_cov_spec'
    
    fig_cov_LD_fname  = pictures_path + 'Participant : ' +meg[1:11] +'LD_noise_cov'
    fig_spec_LD_fname = pictures_path + 'Participant : ' +meg[1:11] +'LD_noise_cov_spec'
    
    
    fig_cov_SD, fig_spectra_SD = mne.viz.plot_cov(cov_SD, epochs_SD.info)
    fig_cov_SD.savefig(fig_cov_SD_fname)
    fig_spectra_SD.savefig(fig_spec_SD_fname)    
    
    fig_cov_LD, fig_spectra_LD = mne.viz.plot_cov(cov_LD, epochs_LD.info)
    fig_cov_LD.savefig(fig_cov_LD_fname)
    fig_spectra_LD.savefig(fig_spec_LD_fname) 
    plt.close('all')