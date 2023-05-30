#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:59:30 2020

@author: sr05
"""

import os
import pickle
import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse_epochs, apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity,seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from mne.stats import (permutation_cluster_1samp_test,spatio_temporal_cluster_test,
                       summarize_clusters_stc,permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
import time
import sys

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr_epoch
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
n_subjects = len(subjects)
stc_total_SD=[[[0]*4 for k in range(6)] for w in range(2)]
stc_total_LD=[[[0]*4 for k in range(6)] for w in range(2)]
method='coh'

def SN_functional_connectivity_bands(i,method):           
    s=time.time()
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_equalized_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_equalized_bands_LD_sub'+str(i)+'.json'
    # stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'bl_bands_SD_sub'+str(i)+'.json'
    # stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'bl_bands_LD_sub'+str(i)+'.json'

    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
        

    # Reading epochs
    epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
    epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
    
    epochs_SD = epochs_sd['words'].copy().resample(500)
    epochs_LD = epochs_ld['words'].copy().resample(500)
    
    equalize_epoch_counts([epochs_SD, epochs_LD])
    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_SD = read_inverse_operator(inv_fname_SD) 
    inv_op_LD = read_inverse_operator(inv_fname_LD) 
                
    stc_sd = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                          pick_ori="normal", return_generator=False)
    stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                            pick_ori="normal", return_generator=False)
    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']
    # Construct indices to estimate connectivity between the label time course
    # and all source space time courses
    vertices_SD = [src_SD[j]['vertno'] for j in range(2)]
    n_signals_tot = 1 + len(vertices_SD[0]) + len(vertices_SD[1])
    indices = seed_target_indices([0], np.arange(1, n_signals_tot))
    
    morph_SD = mne.compute_source_morph(src=inv_op_SD['src'],\
                    subject_from=sub_to, subject_to=C.subject_to,\
                    spacing=C.spacing_morph, subjects_dir=C.data_path) 
            
    morph_LD = mne.compute_source_morph(src= inv_op_LD['src'],\
                    subject_from=sub_to, subject_to=C.subject_to,\
                    spacing=C.spacing_morph, subjects_dir=C.data_path) 
   
    

    for win in np.arange(0, len(C.con_time_window)-1):
        print('[i,win]: ',i,win)
        
        t_min = C.con_time_window[win]
        t_max = C.con_time_window[win+1]
        stc_SD=[]
        stc_LD=[]
        
        
        for n in np.arange(0,len(stc_sd)):
              stc_SD.append(stc_sd[n].copy().crop(t_min*1e-3,t_max*1e-3))
        
        for n in np.arange(0,len(stc_ld)):
              stc_LD.append(stc_ld[n].copy().crop(t_min*1e-3,t_max*1e-3))

        for k in np.arange(0,6):
            print('[i,win,k]: ',i,win,k)
            morphed_labels[k].name = C.rois_labels[k]
    
            seed_ts_sd = mne.extract_label_time_course(stc_SD, morphed_labels[k], \
                        src_SD, mode='mean_flip',return_generator=False)
            seed_ts_ld = mne.extract_label_time_course(stc_LD, morphed_labels[k], \
                        src_LD, mode='mean_flip',return_generator=False)
         
            for f in np.arange(0,len(C.con_freq_band)-1):
                print('[i,win,k,f]: ',i,win,k,f)
                f_min=C.con_freq_band[f]
                f_max=C.con_freq_band[f+1]
                print(f_min,f_max)

                comb_ts_sd = zip(seed_ts_sd, stc_SD)
                comb_ts_ld = zip(seed_ts_ld, stc_LD)  
                
                con_SD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    comb_ts_sd, method=method, mode='fourier', indices=indices,
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
                                    
                con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    comb_ts_ld, method=method, mode='fourier', indices=indices,
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
                    
                con_stc_SD = mne.SourceEstimate(con_SD, vertices=vertices_SD,\
                              tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)
                    
                con_stc_LD = mne.SourceEstimate(con_LD, vertices=vertices_SD,\
                              tmin=t_min*1e-3, tstep=2e-3,subject=sub_to)    
                
        
                stc_total_SD[win][k][f]= morph_SD.apply(con_stc_SD)
                stc_total_LD[win][k][f]= morph_LD.apply(con_stc_LD)

    with open(stc_SD_file_name, "wb") as fp:   #Pickling
        pickle.dump(stc_total_SD, fp)
        
    with open(stc_LD_file_name, "wb") as fp:   #Pickling
        pickle.dump(stc_total_LD, fp)
    e=time.time()   
    print(e-s)
    
    
    
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(C.subjects)) 
    # sbj_id = np.array([3,5,9,10,11,12,16,17])


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]    

for s in sbj_ids:
    SN_functional_connectivity_bands(s,method)



    # stc_SD_bl=[]
    # stc_LD_bl=[]
    # for n in np.arange(0,len(stc_sd)):
    #           stc_SD_bl.append(stc_sd[n].copy().crop(-200*1e-3,0*1e-3))

        
    # for n in np.arange(0,len(stc_ld)):
    #           stc_LD_bl.append(stc_ld[n].copy().crop(-200*1e-3,0*1e-3))


    # for k in np.arange(0,6):
    #     print('[i,k]: ',i,k)
    #     morphed_labels[k].name = C.ROIs_lables[k] 

    #     seed_ts_sd = mne.extract_label_time_course(stc_SD_bl, morphed_labels[k], \
    #                src_SD, mode='mean_flip',return_generator=False)
    #     seed_ts_ld = mne.extract_label_time_course(stc_LD_bl, morphed_labels[k], \
    #                src_LD, mode='mean_flip',return_generator=False)
     
    #     for f in np.arange(0,len(C.con_freq_band)-1):
    #         print('[i,k,f]: ',i,k,f)
    #         f_min=C.con_freq_band[f]
    #         f_max=C.con_freq_band[f+1]
    #         print(f_min,f_max)

    #         comb_ts_sd = zip(seed_ts_sd, stc_SD_bl)
    #         comb_ts_ld = zip(seed_ts_ld, stc_LD_bl)  
            
    #         con_SD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    #             comb_ts_sd, method=method, mode='fourier', indices=indices,
    #             sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
                                
    #         con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    #             comb_ts_ld, method=method, mode='fourier', indices=indices,
    #             sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
                
    #         con_stc_SD = mne.SourceEstimate(con_SD, vertices=vertices_SD,\
    #                      tmin=-200*1e-3, tstep=2e-3,subject=sub_to)
                
    #         con_stc_LD = mne.SourceEstimate(con_LD, vertices=vertices_SD,\
    #                      tmin=-200*1e-3, tstep=2e-3,subject=sub_to)    
            
    
    #         stc_total_SD[k][f]= morph_SD.apply(con_stc_SD)
    #         stc_total_LD[k][f]= morph_LD.apply(con_stc_LD)

    # with open(stc_SD_file_name, "wb") as fp:   #Pickling
    #     pickle.dump(stc_total_SD, fp)
        
    # with open(stc_LD_file_name, "wb") as fp:   #Pickling
    #     pickle.dump(stc_total_LD, fp)