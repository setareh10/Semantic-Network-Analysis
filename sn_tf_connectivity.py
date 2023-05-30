#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:58:58 2020

@author: sr05
"""


import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from SN_matrix_mirror import matrix_mirror 
from mne.stats import (permutation_cluster_1samp_test,
                       summarize_clusters_stc,permutation_cluster_test)
from scipy import stats as stats
# from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,
                       f_mway_rm, summarize_clusters_stc)
from matplotlib import pyplot as plt
import statsmodels.stats.multicomp as multi
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
X_SD = np.zeros([len(subjects),len(SN_ROI),250])
X_LD = np.zeros([len(subjects),len(SN_ROI),250])
times = np.arange(-300,901)


con_SD_TF=np.zeros([18,6,6,18,376])
con_LD_TF=np.zeros([18,6,6,18,376])

  
for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]

    # print('Participant : ' , i, '/ win : ',win, '/ freq : ',freq)
    print('Participant : ' , i)

    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
        

    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'

    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

    epochs_f= mne.epochs.combine_event_ids(epochs_fruit,['visual',
                         'hear','hand','neutral','emotional'], {'words':15})
    epochs_o= mne.epochs.combine_event_ids(epochs_odour,['visual',
                         'hear','hand','neutral','emotional'], {'words':15})    
    epochs_m= mne.epochs.combine_event_ids(epochs_milk,['visual',
                         'hear','hand','neutral','emotional'], {'words':15})
    
    epochs_f=epochs_f['words'].copy().crop(-.200,0.550).resample(500)
    epochs_o=epochs_o['words'].copy().crop(-.200,0.550).resample(500)
    epochs_m=epochs_m['words'].copy().crop(-.200,0.550).resample(500)   
    epochs_LD = epochs_ld['words'].copy().crop(-.200,0.550).resample(500)


    
    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_SD = read_inverse_operator(inv_fname_SD) 
    inv_op_LD = read_inverse_operator(inv_fname_LD) 
                
    stc_f = apply_inverse_epochs(epochs_f, inv_op_SD,lambda2,method ='MNE', 
                      pick_ori="normal", return_generator=False)
    stc_o = apply_inverse_epochs(epochs_o, inv_op_SD,lambda2,method ='MNE', 
                          pick_ori="normal", return_generator=False)
    stc_m = apply_inverse_epochs(epochs_m, inv_op_SD,lambda2,method ='MNE', 
                      pick_ori="normal", return_generator=False)

    stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                            pick_ori="normal", return_generator=False)

    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']

 
 

    for k in np.arange(0,6):
            morphed_labels[k].subject = sub_to  
            
    labels_ts_f = mne.extract_label_time_course(stc_f, morphed_labels, \
                   src_SD, mode='mean_flip',return_generator=False)
    labels_ts_o = mne.extract_label_time_course(stc_o, morphed_labels, \
               src_SD, mode='mean_flip',return_generator=False)
    labels_ts_m = mne.extract_label_time_course(stc_m, morphed_labels, \
               src_SD, mode='mean_flip',return_generator=False)

    labels_ts_ld= mne.extract_label_time_course(stc_ld, morphed_labels, \
           src_LD, mode='mean_flip',return_generator=False)
      
    # label_SD=np.zeros([len(label_sd),6,550])
    # label_LD=np.zeros([len(label_ld),6,550])

    # for m in np.arange(0,len(label_sd)):
    #     label_SD[m,:,:]=label_sd[m][:,300:850]
    #     label_LD[m,:,:]=label_ld[m][:,300:850]

    print('Participant : ' , i, ' SD Connectivity')

    con_F,freqs_LD,times_LD,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                labels_ts_f, method='ppc', mode='cwt_morlet',\
                sfreq=C.sfreq, fmin=4, fmax=40, n_jobs=10,cwt_freqs=\
                np.arange(4,40,2),cwt_n_cycles=5)
    con_O,freqs_LD,times_LD,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                labels_ts_o, method='ppc', mode='cwt_morlet',\
                sfreq=C.sfreq, fmin=4, fmax=40, n_jobs=10,cwt_freqs=\
                np.arange(4,40,2),cwt_n_cycles=5)
    con_M,freqs_LD,times_m,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                labels_ts_ld, method='ppc', mode='cwt_morlet',\
                sfreq=C.sfreq, fmin=4, fmax=40, n_jobs=10,cwt_freqs=\
                np.arange(4,40,2),cwt_n_cycles=5)
               
    print('Participant : ' , i, ' LD Connectivity')

    con_LD,freqs_LD,times_LD,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                labels_ts_ld, method='ppc', mode='cwt_morlet',\
                sfreq=C.sfreq, fmin=4, fmax=40, n_jobs=10,cwt_freqs=\
                np.arange(4,40,2),cwt_n_cycles=5)
        
    con_SD_TF[i,:,:,:,:] = (con_F+ con_O+ con_M)/3 
    con_LD_TF[i,:,:,:,:] = con_LD  

###############################################################################

ROI_label=['lATL','rATL','TG','IFG','AG','PVA']               
for i in np.arange(0,6):
    for j in np.arange(i+1,6):
        print(ROI_label[i],ROI_label[j])
        X=con_SD_TF.copy().mean(0)[j,i,:,:]
        Y=con_LD_TF.copy().mean(0)[j,i,:,:]
        vmax=max(X.max(),Y.max())
        vmin=min(X.min(),Y.min())

        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(X, cmap=plt.cm.hot,
                    extent=[-200, 550, 4, 40],vmin=vmin,vmax=vmax,
                    aspect='auto', origin='lower')
        plt.title(ROI_label[i]+'-'+ROI_label[j])
        plt.colorbar()
        
        plt.subplot(2,1,2)
        plt.imshow(Y, cmap=plt.cm.hot,vmin=vmin,vmax=vmax,
                    extent=[-200, 550, 4, 40],
                    aspect='auto', origin='lower')
        plt.colorbar()
# plt.close('all')

###############################################################################
## t-test and cluster-based correction for each ROI
# c=0
# X=np.zeros([18,15,18,550])
# Y=np.zeros([18,15,18,550])
# n_permutations=5000
# p_threshold=0.05
# t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)

# for i in np.arange(0,6):
#     for j in np.arange(i+1,6):
#         X[:,c,:,:]=con_SD_TF[:,j,i,:,:]
#         Y[:,c,:,:]=con_LD_TF[:,j,i,:,:]
#         c=c+1
# Z=X-Y
# for k in np.arange(0,15):
#     T_obs, clusters, cluster_p_values, H0 = \
#         permutation_cluster_1samp_test(Z[:,k,:,:], n_permutations=n_permutations,
#                                         threshold=t_threshold, tail=0,
#                                         connectivity=None,out_type='mask',
#                                         verbose=True)
        
#     T_obs_plot = np.nan * np.ones_like(T_obs)
#     for c, p_val in zip(clusters, cluster_p_values):
#         if p_val <= 0.05:
#             T_obs_plot[c] = T_obs[c]
            
#     T_obs_ttest = np.nan * np.ones_like(T_obs)
#     for r in np.arange(0,18):
#         for c in np.arange(0,550):
#             if abs(T_obs[r,c])>t_threshold:
#                 T_obs_ttest[r,c] =  T_obs[r,c]
            
#     vmax = np.max(T_obs)
#     vmin = np.min(T_obs)
#     plt.figure()
    
#     plt.subplot(311)
#     plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
#                 extent=[0, 550,4, 40],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()
#     plt.ylabel('Frequency (Hz)')
#     # plt.title('TFR of '+lb[k])
    
    
#     plt.subplot(312)
#     plt.imshow(T_obs, cmap=plt.cm.bone,
#                 extent=[0, 550,4, 40],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
#     plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
#                 extent=[0, 550,4, 40],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()

#     plt.ylabel('Frequency (Hz)')

    
#     plt.subplot(313)
#     plt.imshow(T_obs, cmap=plt.cm.gray,label='cluster-based permutation test',
#                 extent=[0, 550,4, 40],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
#     plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,label='cluster-based permutation test',
#                 extent=[0, 550,4, 40],
#                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#     plt.colorbar()

#     plt.xlabel('Time (ms)')
#     plt.ylabel('Frequency (Hz)')

#     plt.show()      
#     # plt.savefig(C.pictures_path_Source_estimate+ 't-test_TFR_'+lb[k]+'.png')

    
# # plt.close('all') 