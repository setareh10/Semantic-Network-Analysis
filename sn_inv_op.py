#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:15:23 2020

@author: sr05
"""


import numpy as np
import mne
from mne.minimum_norm import ( make_inverse_operator, write_inverse_operator)
import sn_config as C

# path to raw data 
data_path = C.data_path
main_path = C.main_path


subjects =  C.subjects    

for i in np.arange(0,len(subjects)):
    print('Participant : ' ,i)
    meg = subjects[i]
    
    # path to raw data 
    raw_fname_SD  = data_path + meg + 'block_SD_tsss_raw.fif'
    raw_fname_LD  = main_path + meg + 'block_LD_tsss_raw.fif'

    
    # loading raw data
    raw_SD = mne.io.Raw(raw_fname_SD , preload=True)
    raw_LD = mne.io.Raw(raw_fname_LD , preload=True)

    
    # path to noise covariance matrix
    cov_fname_SD = data_path + meg + 'block_SD-noise-cov.fif'
    cov_fname_LD = data_path + meg + 'block_LD-noise-cov.fif'

    # Loading noise covariance matrix
    cov_SD = mne.read_cov(cov_fname_SD)
    cov_LD = mne.read_cov(cov_fname_LD)

    
     ##................................EEG + MEG............................##

    pick_eeg, pick_meg = True, True
    
    # path to fwd solution file
    fwd_fname_EMEG = data_path + meg + 'block_EMEG_fwd.fif'

    # Loading fwd solution file
    fwd = mne.read_forward_solution(fwd_fname_EMEG)
    fwd_EMEG = mne.pick_types_forward(fwd , meg=pick_meg, eeg=pick_eeg)

    # make an inverse operator    
    inv_op_SD = make_inverse_operator(raw_SD.info , fwd_EMEG , noise_cov=cov_SD, 
                                      loose=0.2, depth=None, verbose=None)
    inv_op_LD = make_inverse_operator(raw_LD.info , fwd_EMEG , noise_cov=cov_LD, 
                                      loose=0.2, depth=None, verbose=None)
    
    # Path to save inverse operateor
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'


    write_inverse_operator(inv_fname_SD , inv_op_SD)
    write_inverse_operator(inv_fname_LD , inv_op_LD)

     ##................................Only EEG ............................##

    pick_eeg, pick_meg = True, False
    
    fwd_fname_EEG = data_path + meg + 'block_EMEG_fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname_EEG)

    fwd_EEG = mne.pick_types_forward(fwd , meg=pick_meg, eeg=pick_eeg)

    # make an inverse operator    
    inv_op_SD = make_inverse_operator(raw_SD.info , fwd_EEG , noise_cov=cov_SD,
                                       loose=0.2, depth=None, verbose=None)
    inv_op_LD = make_inverse_operator(raw_LD.info , fwd_EEG , noise_cov=cov_LD,
                                      loose=0.2, depth=None, verbose=None)
     
     
    inv_fname_SD = data_path + meg + 'InvOp_SD_EEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EEG-inv.fif'

    write_inverse_operator(inv_fname_SD , inv_op_SD)
    write_inverse_operator(inv_fname_LD , inv_op_LD)

     ##................................Only MEG ............................##
    pick_eeg, pick_meg = False, True
    
    fwd_fname_MEG = data_path + meg + 'block_MEG_fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname_MEG)

    fwd_MEG = mne.pick_types_forward(fwd , meg=pick_meg, eeg=pick_eeg)

    # make an inverse operator    
    
    inv_op_SD = make_inverse_operator(raw_SD.info , fwd_MEG , noise_cov=cov_SD,
                                      loose=0.2, depth=None, verbose=None)
    inv_op_LD = make_inverse_operator(raw_LD.info , fwd_MEG , noise_cov=cov_LD,
                                      loose=0.2, depth=None, verbose=None)
    
    
    inv_fname_SD = data_path + meg + 'InvOp_SD_MEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_MEG-inv.fif'


    write_inverse_operator(inv_fname_SD , inv_op_SD)
    write_inverse_operator(inv_fname_LD , inv_op_LD)