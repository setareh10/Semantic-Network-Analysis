#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:12:47 2020

@author: sr05
"""

import numpy as np
import mne
import sn_config as C
from surfer import Brain

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
from mne.time_frequency import tfr_morlet

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects

l_ATL = mne.read_label(data_path+ 'L_ATL_myROI_lh.label')  
r_ATL = mne.read_label(data_path+ 'R_ATL_myROI_rh.label')  
IFG   = mne.read_label(data_path+ 'L_IFG_myROI_lh.label')   
TG    = mne.read_label(data_path+ 'L_TG_myROI_lh.label')   
AG    = mne.read_label(data_path+ 'L_AG_myROI_lh.label')  
V     = mne.read_label(data_path+ 'L_V1_myROI_lh.label')   
labels = [l_ATL , r_ATL, IFG, TG, AG, V]


for win in np.arange(0, len(C.time_window)-4):
    t_min_crop = C.time_window[win]
    t_max_crop = C.time_window[win]+ C.time_window_len
    X = np.zeros([len(subjects),12])
    Y = np.zeros([len(subjects),12,101])

    for i in np.arange(0, len(subjects)-17):
        n_subjects = len(subjects)
        meg = subjects[i]
        print('Participant : ' , i, '/ win : ',win)
        
        # Reading epochs
        epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
        epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
            
        epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
        epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

        epochs_SD = epochs_sd['words'] 
        epochs_LD = epochs_ld['words'] 
    
        # Equalize trial counts to eliminate bias (which would otherwise be
        # introduced by the abs() performed below)
        equalize_epoch_counts([epochs_SD, epochs_LD])
        
        # Reading inverse operator
        inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
        inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
    
        inv_op_SD = read_inverse_operator(inv_fname_SD) 
        inv_op_LD = read_inverse_operator(inv_fname_LD) 
        
        # Evoked responses 
        evoked_SD = epochs_SD.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
        evoked_LD = epochs_LD.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
                    
        stc_SD = apply_inverse( evoked_SD, inv_op_SD,lambda2,method ='MNE', 
                              pick_ori="normal")
        stc_LD = apply_inverse( evoked_LD, inv_op_LD,lambda2,method ='MNE',
                                pick_ori="normal")
        
        stc_SD1 = stc_SD.copy().crop(t_min_crop, t_max_crop)
        stc_LD1 = stc_LD.copy().crop(t_min_crop, t_max_crop)

        src_SD = inv_op_SD['src']
        src_LD = inv_op_LD['src']
        # Average the source estimates within each label using sign-flips to reduce
        # signal cancellations, also here we return a generator
         
        label_ts_SD = mne.extract_label_time_course(stc_SD1, labels, src_SD,\
                      mode='mean_flip')       
        label_ts_LD = mne.extract_label_time_course(stc_LD1, labels, src_LD,\
                      mode='mean_flip')  
        
        # Averaging sourse signals across a time window :[0.050:0.100:0.550]
        stc_SD_mean = label_ts_SD
        stc_LD_mean = label_ts_LD
        

        
        
        # X[i,0:6]  = stc_SD_mean
        # X[i,6:12] = stc_LD_mean
        # Y[i,0:6,:]  =  label_ts_SD 
        # Y[i,6:12,:] =  label_ts_LD 
    # fval , pval = mne.stats.f_mway_rm(X,factor_levels =[2,6],effects ='all',\
    #                                   correction=False, return_pvals=True)
    
    # fval , pval = mne.stats.f_mway_rm(Y,factor_levels =[2,6],effects ='all',\
    #                                   correction=False, return_pvals=True)
        
        