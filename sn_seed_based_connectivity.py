#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:47:12 2020

@author: sr05
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:43 2020

@author: sr05
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse_epochs, apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity,seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from SN_matrix_mirror import matrix_mirror 
from mne.stats import (permutation_cluster_1samp_test,spatio_temporal_cluster_test,
                       summarize_clusters_stc,permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
from SN_label_baseline_correction import label_baseline_correction

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
SN_ROI_name=['lATL','rATL','TG','IFG','AG','PVA']
X_SD = np.zeros([len(subjects),len(SN_ROI),250])
X_LD = np.zeros([len(subjects),len(SN_ROI),250])
times = np.arange(-300,901)



for win in np.arange(0, len(C.con_time_window)-1):
    tmin = C.con_time_window[win]
    tmax = C.con_time_window[win]+ C.con_time_window_len
    # for freq in np.arange(0, len(C.con_freq_band)-3):
    #     fmin = C.con_freq_band[freq]
    #     fmax = C.con_freq_band[freq+1]     
    for i in np.arange(0, len(subjects)-17):
        n_subjects = len(subjects)
        meg = subjects[i]
        sub_to = MRI_sub[i][1:15]
    
        # print('Participant : ' , i, '/ win : ',win, '/ freq : ',freq)
        print('Participant : ' , i)
    
        morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                      subject_from='fsaverage',subjects_dir=data_path)
            
    
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
                    
        stc_sd = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                              pick_ori="normal", return_generator=False)
        stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                                pick_ori="normal", return_generator=False)
        times=epochs_sd.times
        stc_SD =[]
        stc_LD =[]

        for n in np.arange(0,len(stc_sd)):
            stc_SD.append(stc_baseline_correction(stc_sd[n],times))
            stc_LD.append(stc_baseline_correction(stc_ld[n],times))

    
          # Morphing source signals onto fsaverage
        morph_SD = mne.compute_source_morph( src= inv_op_SD['src'],subject_from\
                   = stc_sd[0].subject , subject_to = C.subject_to , spacing = \
                   C.spacing_morph, subjects_dir = C.data_path)    
        morph_LD = mne.compute_source_morph( src= inv_op_LD['src'],subject_from\
                   = stc_ld[0].subject , subject_to = C.subject_to , spacing = \
                   C.spacing_morph, subjects_dir = C.data_path) 
        
        stc_fsaverage_SD=[]
        stc_fsaverage_LD=[]

        for n in np.arange(0,len(stc_SD)):
            stc_fsaverage_SD.append(morph_SD.apply(stc_SD[n]))
            stc_fsaverage_LD.append(morph_LD.apply(stc_LD[n]))

    
        # stc_fsaverage_SD=[morph_SD.apply(stc) for stc in stc_sd]
        # stc_fsaverage_LD=[morph_LD.apply(stc) for stc in stc_ld]
        
    
        src_SD = inv_op_SD['src']
        src_LD = inv_op_LD['src']
        
        for k in np.arange(0,1):
            morphed_labels[k].name = SN_ROI_name[k] 

            seed_ts_sd = mne.extract_label_time_course(stc_SD, morphed_labels[k], \
                       src_SD, mode='mean_flip',return_generator=False)
            seed_ts_ld = mne.extract_label_time_course(stc_LD, morphed_labels[k], \
                       src_LD, mode='mean_flip',return_generator=False)
           
       
            # seed_ts_SD=[]
            # seed_ts_LD=[]
            
            # for n in np.arange(0,len(stc_SD)):

            #     seed_ts_SD.append(mne.SourceEstimate(seed_ts_sd[n], vertices=vertices, tstep=1,
            #                  subject=sub_to) ) 
                
            #     seed_ts_LD.append(mne.SourceEstimate(seed_ts_ld[n], vertices=vertices, tstep=1,
            #              subject=sub_to))  
            comb_ts_sd = zip(seed_ts_sd, stc_fsaverage_SD)
            comb_ts_ld = zip(seed_ts_ld, stc_fsaverage_LD)

            # Construct indices to estimate connectivity between the label time course
            # and all source space time courses
            # vertices_SD = [src_SD[j]['vertno'] for j in range(2)]
            # vertices_LD = [src_LD[j]['vertno'] for j in range(2)]
            vertices = [np.arange(10242), np.arange(10242)]

            n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])
            
            indices = seed_target_indices([0], np.arange(1, n_signals_tot))

            fmin = 8.
            fmax = 12.
            tmin_con = 0.
            sfreq = C.sfreq  # the sampling frequency
            
            psi, freqs, times, n_epochs, n_tapers = phase_slope_index(
                comb_ts_sd,indices=indices, sfreq=sfreq, mode='fourier', 
                fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_jobs=6)
                
            psi_stc = mne.SourceEstimate(psi, vertices=vertices, tmin=0, tstep=1,
                         subject=sub_to)

            # Now we can visualize the PSI using the :meth:`~mne.SourceEstimate.plot`
            # method. We use a custom colormap to show signed values
            v_max = np.max(np.abs(psi))
            brain = psi_stc.plot(surface='inflated', hemi='split',
                                 time_label='Phase Slope Index (PSI)',
                                 subjects_dir=data_path,size=([800,400]),
                                 clim=dict(kind='percent', pos_lims=( 10,50,100)))
            brain.show_view('lateral')
            # brain.add_label(morphed_labels[k], color='green', alpha=0.7)
                  
            # THIS IS A COMMENT