#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:33:02 2020

@author: sr05
"""

import mne
import numpy as np
import sn_config as C
from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from scipy import stats

def stc_baseline_correction(X):
    time_dim = len(X.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    baseline_timepoints = X.times[0:300]

    baseline_mean = X.data[:,0:len(baseline_timepoints)].mean(1)

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean),1]),\
                                  time_dim  ,axis=1)
    corrected_stc = X - baseline_mean_mat
    return corrected_stc

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2
X1 = np.zeros([len(subjects),20484 ])
X2 = np.zeros([len(subjects),20484 ])
for w in np.arange(0,len(C.time_window)):
    
    t_min_crop= C.time_window[w]
    t_max_crop= C.time_window[w] + C.time_window_len
    
    for i in np.arange(0, len(subjects)):
        
        n_subjects = len(subjects)
        meg = subjects[i]
        print('Participant : ' , i, '/win : ',C.time_window[w])
        
        # Reading epochs
        epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
        epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
            
        epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
        epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
        
        epochs_SD = epochs_sd['words'] 
        epochs_LD = epochs_ld['words'] 
    
        # Equalize trial counts to eliminate bias (which would otherwise be
        # introduced by the abs() performed below)
        # equalize_epoch_counts([epochs_SD, epochs_LD])
        
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
        
        # Applying inverse solution to get sourse signals    
        stc_sd = apply_inverse(evoked_SD, inv_op_SD,lambda2,method ='MNE', 
                               pick_ori=None)
        stc_ld = apply_inverse(evoked_LD, inv_op_LD,lambda2,method ='MNE',
                               pick_ori=None)
        stc_sd_corrected = stc_baseline_correction(stc_sd ) 
        stc_ld_corrected = stc_baseline_correction(stc_ld ) 

        
        # Averaging sourse signals across a time window :[0.050:0.100:0.550]
        stc_SD = stc_sd_corrected.copy().crop(t_min_crop, t_max_crop)
        stc_LD = stc_ld_corrected.copy().crop(t_min_crop, t_max_crop)
        tmin = stc_SD.tmin
        tstep = stc_SD.tstep

        # Morphing source signals onto fsaverage
        morph_SD = mne.compute_source_morph( src= inv_op_SD['src'],subject_from\
                    = stc_SD.subject , subject_to = C.subject_to , spacing = \
                    C.spacing_morph, subjects_dir = C.data_path)    
        morph_LD = mne.compute_source_morph( src= inv_op_LD['src'],subject_from\
                    = stc_LD.subject , subject_to = C.subject_to , spacing = \
                    C.spacing_morph, subjects_dir = C.data_path) 
        
        stc_fsaverage_SD = morph_SD.apply(stc_SD)
        stc_fsaverage_LD = morph_LD.apply(stc_LD)
        stc = stc_fsaverage_SD - stc_fsaverage_LD
        n_vertices_sample, n_times = stc.data.shape

        # Defining two conditions as X1 and X2
     
        X1[i, :] = stc_fsaverage_SD.copy().data.mean(1)
        X2[i, :] = stc_fsaverage_LD.copy().data.mean(1)
     
    
    # Calculate the t-test on TWO RELATED samples of scores, X1 and X2
    t_value , p_value = stats.ttest_rel(X1,X2)
    # alpha= 0.055
    # reject_fdr, pval_fdr = mne.stats.fdr_correction(p_value, alpha=alpha, method='indep')
    # threshold_fdr = np.min(np.abs(t_value)[reject_fdr])
    vertices_to = [np.arange(10242), np.arange(10242)]

    # Visualizing
    tval_stc2 = mne.SourceEstimate(t_value, vertices=vertices_to,
                tmin=t_min_crop,  tstep= tstep , subject='fsaverage')

    C.stc_all.append(tval_stc2)
    idx = tval_stc2.time_as_index(times=tval_stc2.times)
    data = tval_stc2.data[:, idx]
    C.min_max_val.append([data.min() , data.max()])


max_val = max(max(C.min_max_val))
min_val = min(min(min(C.min_max_val)),0)
thresh = max(abs(max_val), abs(min_val))
mid_val = (max_val  + min_val)/5      
                    
# for n in np.arange(0, len(C.stc_all)): 
#     t_min_crop= C.time_window[n]
#     t_max_crop= C.time_window[n] + C.time_window_len
    
#     # brain = C.stc_all[n].plot(surface='inflated', hemi='split',subject =\
#     #       'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', pos_lims=\
#     #       [2.10,2.11,7.68] ),size=(800,400))
#     # brain.save_image(C.pictures_path_Source_estimate+'Thresholded_T_maps_'+\
#     #                   f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}_unequalized.png')
 

#     brain = C.stc_all[n].plot(surface='inflated', views='caudal',hemi='both',subject =\
#           'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', pos_lims=\
#           [2.10,2.11,7.68] ),size=(800,400))  

#     brain.save_image(C.pictures_path_Source_estimate+'Thresholded_T_maps_caudal_'+\
#                       f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}_unequalized.png')
 



