#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:33:02 2020

@author: sr05
"""

import mne
import numpy as np
import sn_config as C
# import my_baseline_correction 

from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from scipy import stats as stats
# from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator


# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = '/imaging/rf02/TypLexMEG/fsaverage/label'
# X = np.zeros([20484, 1, 18, 2])
# X = np.zeros([20484,20, len(subjects), 2])

def my_baseline_correction(X):
    time_dim = len(X.times)
    baseline_timepoints = X.times[np.where(X.times<0)]
    axis0,axis1 = (X.data).shape
    baseline_mean = X.data[:,0:len(baseline_timepoints)].mean(1)
    if axis0==time_dim:
        axis = 0
    else:
        axis = 1
    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean),1]),\
                                  time_dim  ,axis=axis )
    corrected_stc = X - baseline_mean_mat
    return corrected_stc
  
    
for win in np.arange(4, len(C.time_window)):

    t_min_crop= C.time_window[win]
    t_max_crop= C.time_window[win] + C.time_window_len
    X = np.zeros([20484,51, len(subjects), 5])

    for i in np.arange(0, len(subjects)):
        n_subjects = len(subjects)
        meg = subjects[i]
        print('Participant : ' , i, '/ win : ',win)
        
        # Reading epochs
        epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'     
        epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
        epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
        epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
        epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
        
        epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
        epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
        epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)

        epochs_f = mne.epochs.combine_event_ids(epochs_fruit,['visual',
                              'hear','hand','neutral','emotional'], {'words':15})
        epochs_o = mne.epochs.combine_event_ids(epochs_odour,['visual',
                              'hear','hand','neutral','emotional'], {'words':15})    
        epochs_m  = mne.epochs.combine_event_ids(epochs_milk,['visual',
                              'hear','hand','neutral','emotional'], {'words':15})
        
        
            
        epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
        epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
        
        epochs_SD = epochs_sd['words'] 
        epochs_LD = epochs_ld['words'] 
    
       
        # Reading inverse operator
        inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
        inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
    
        inv_op_SD = read_inverse_operator(inv_fname_SD) 
        inv_op_LD = read_inverse_operator(inv_fname_LD) 

        
        # Evoked responses 
        evoked_sd = epochs_SD.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
        evoked_f = epochs_f.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
        evoked_o = epochs_o.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
        evoked_m = epochs_m.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
                                                
        evoked_ld = epochs_LD.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
            
            
        evoked_SD = evoked_sd.copy().resample(500)
        evoked_LD = evoked_ld.copy().resample(500)
        
        evoked_f = evoked_f.copy().resample(500)
        evoked_o = evoked_o.copy().resample(500)
        evoked_m = evoked_m.copy().resample(500)

        # evoked_SD = evoked_sd.copy()
        # evoked_LD = evoked_ld.copy()

        # Applying inverse solution to get sourse signals    
        stc_sd = apply_inverse(evoked_SD, inv_op_SD,lambda2,method ='MNE', 
                               pick_ori=None)
        stc_f = apply_inverse(evoked_f, inv_op_SD,lambda2,method ='MNE', 
                               pick_ori=None)
        stc_o = apply_inverse(evoked_o, inv_op_SD,lambda2,method ='MNE', 
                               pick_ori=None)
        stc_m = apply_inverse(evoked_m, inv_op_SD,lambda2,method ='MNE', 
                               pick_ori=None)
                                
        stc_ld = apply_inverse(evoked_LD, inv_op_LD,lambda2,method ='MNE',
                               pick_ori=None)
        stc_sd_corrected = my_baseline_correction(stc_sd ) 
        stc_f_corrected = my_baseline_correction(stc_f ) 
        stc_o_corrected = my_baseline_correction(stc_o ) 
        stc_m_corrected = my_baseline_correction(stc_m ) 

        stc_ld_corrected = my_baseline_correction(stc_ld ) 

        
        # Averaging sourse signals across a time window :[0.050:0.100:0.550]
        stc_SD = stc_sd_corrected.copy().crop(t_min_crop, t_max_crop)
        stc_F = stc_f_corrected.copy().crop(t_min_crop, t_max_crop)
        stc_O = stc_o_corrected.copy().crop(t_min_crop, t_max_crop)
        stc_M = stc_m_corrected.copy().crop(t_min_crop, t_max_crop)

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
        stc_fsaverage_f = morph_SD.apply(stc_F)
        stc_fsaverage_o = morph_SD.apply(stc_O)
        stc_fsaverage_m = morph_SD.apply(stc_M)

        stc_fsaverage_LD = morph_LD.apply(stc_LD)
        stc = stc_fsaverage_SD - stc_fsaverage_LD
        n_vertices_sample, n_times = stc.data.shape

        # # X[:, win, i, 0] = stc_fsaverage_SD.data.mean(1).reshape(20484)
        # X[:, win, i, 1] = stc_fsaverage_LD.data.mean(1).reshape(20484)
        X[:, :, i, 0] = stc_fsaverage_SD.data
        X[:, :, i, 1] = stc_fsaverage_f.data
        X[:, :, i, 2] = stc_fsaverage_o.data
        X[:, :, i, 3] = stc_fsaverage_m.data

        X[:, :, i, 4] = stc_fsaverage_LD.data

    # Subtarcting 2 condistions
    Y = X[:, :, :, 0] - X[:, :, :, 4]  # make paired contrast
    # Defining Y as an rray of shape: observations(subjects) x time x vertices(space)
    Y = np.transpose(Y, [2, 1, 0])
    
#    fname_label = label_path + '/' + 'toremove_wbspokes-lh.label'; 
    fname_label = '/imaging/hauk/users/rf02/TypLexMEG/fsaverage/label/toremove_wbspokes-lh.label'
    labelL = mne.read_label(fname_label)
    fname_label = '/imaging/hauk/users/rf02/TypLexMEG/fsaverage/label/toremove_wbspokes-rh.label'
    labelR = mne.read_label(fname_label)
    labelss=labelL+labelR
    bb=stc_SD.in_label(labelss)
    fsave_vertices = [np.arange(10242), np.arange(10242)]
    nnl=np.in1d(fsave_vertices[0],bb.lh_vertno)
    nnr=np.in1d(fsave_vertices[1],bb.rh_vertno)
    spatial_exclude=np.hstack((fsave_vertices[0][nnl], fsave_vertices[0][nnr]+10242))
           
        
        
    # # # adjacency = mne.spatial_src_adjacency(inv_op_SD['src'])
    source_space = mne.grade_to_tris(5)
    # as we only have one hemisphere we need only need half the connectivity
    print('Computing connectivity.')
    connectivity = mne.spatial_tris_connectivity(source_space)
    p_threshold = 0.05
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
       
    #     print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
          spatio_temporal_cluster_1samp_test(Y, connectivity= connectivity,\
          n_jobs=10,threshold=t_threshold,n_permutations=5000,spatial_exclude =\
          spatial_exclude,step_down_p=0.05,t_power=1,tail=0)


       
    # print('Visualizing clusters.')
         
    fsave_vertices = [np.arange(10242),np.arange(10242)]
    
    #     # Build a convenient representation of each cluster, where each
    #     # cluster becomes a "time point" in the SourceEstimate
    stc_all_cluster_vis = summarize_clusters_stc(clu,tstep=tstep*1000,\
                                                  vertices = fsave_vertices)
     
    idx = stc_all_cluster_vis.time_as_index(times=stc_all_cluster_vis.times)
    data = stc_all_cluster_vis.data[:, idx]
    thresh = max(([abs(data.min()) , abs(data.max())]))
    
    brain = stc_all_cluster_vis.plot(surface='inflated', hemi='split',subject =\
        'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', pos_lims=\
          [0,1,101]),size=(800,400),colormap='mne')
   
    # brain.save_image(C.pictures_path_Source_estimate+'Clusters_100_'+\
    #             f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}_unequalized.png')
   