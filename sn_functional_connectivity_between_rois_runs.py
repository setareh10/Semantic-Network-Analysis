#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 07:33:10 2020

@author: sr05
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""
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
import time
import pickle
import sys
import os
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
con_labels_SD=[[0]*4 for w in range(2)]
con_labels_F=[[0]*4 for w in range(2)]
con_labels_O=[[0]*4 for w in range(2)]
con_labels_M=[[0]*4 for w in range(2)]
con_labels_LD=[[0]*4 for w in range(2)]
method='imcoh'


def SN_functional_connectivity_betweenROIs_runs(i,method):           
    s=time.time()
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    
    stc_F_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_F_bands_SD_sub'+str(i)+'.json'
    stc_O_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_O_bands_LD_sub'+str(i)+'.json'
    stc_M_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_M_bands_SD_sub'+str(i)+'.json'
    stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_mean_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_mean_bands_LD_sub'+str(i)+'.json'

    morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                  subject_from='fsaverage',subjects_dir=data_path)
        

    # Reading epochs
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
    
    epochs_f=epochs_f['words'].copy().resample(500)
    epochs_o=epochs_o['words'].copy().resample(500)
    epochs_m=epochs_m['words'].copy().resample(500)   
    epochs_LD = epochs_ld['words'].copy().resample(500)


    
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

 
 
    for win in np.arange(0, len(C.con_time_window)-1):
        print('[i,win]: ',i,win)
        
        t_min = C.con_time_window[win]
        t_max = C.con_time_window[win+1]
        stc_F=[]
        stc_O=[]
        stc_M=[]
        stc_LD=[]
        
        for n in np.arange(0,len(stc_f)):
              stc_F.append(stc_f[n].copy().crop(t_min*1e-3,t_max*1e-3))
        for n in np.arange(0,len(stc_o)):
              stc_O.append(stc_o[n].copy().crop(t_min*1e-3,t_max*1e-3))
        for n in np.arange(0,len(stc_m)):
              stc_M.append(stc_m[n].copy().crop(t_min*1e-3,t_max*1e-3))
      
        for n in np.arange(0,len(stc_ld)):
             stc_LD.append(stc_ld[n].copy().crop(t_min*1e-3,t_max*1e-3))

        for k in np.arange(0,6):
            # print('[i,win,k]: ',i,win,k)
            morphed_labels[k].name = C.rois_labels[k]
    
        labels_ts_f = mne.extract_label_time_course(stc_F, morphed_labels, \
                   src_SD, mode='mean_flip',return_generator=False)
        labels_ts_o = mne.extract_label_time_course(stc_O, morphed_labels, \
                   src_SD, mode='mean_flip',return_generator=False)
        labels_ts_m = mne.extract_label_time_course(stc_M, morphed_labels, \
                   src_SD, mode='mean_flip',return_generator=False)
    
        labels_ts_ld= mne.extract_label_time_course(stc_LD, morphed_labels, \
               src_LD, mode='mean_flip',return_generator=False)
      
        for f in np.arange(0,len(C.con_freq_band)-1):
            print('[i,win,k,f]: ',i,win,k,f)
            f_min=C.con_freq_band[f]
            f_max=C.con_freq_band[f+1]
            print(f_min,f_max)
 
            
            con_F, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                 labels_ts_f, method=method, mode='fourier', 
                sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
            con_O, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                 labels_ts_o, method=method, mode='fourier', 
                sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
            con_M, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                 labels_ts_m, method=method, mode='fourier', 
                sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
                                              
            con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                 labels_ts_ld, method=method, mode='fourier', 
                sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
            con_SD=(con_F+ con_O+ con_M)/3
            
            con_labels_SD[win][f]= con_SD.reshape(6,6)
            con_labels_LD[win][f]= con_LD.reshape(6,6)
            con_labels_F[win][f]= con_F.reshape(6,6)
            con_labels_O[win][f]= con_O.reshape(6,6)
            con_labels_M[win][f]= con_M.reshape(6,6)

    with open(stc_SD_file_name, "wb") as fp:   #Pickling
        pickle.dump(con_labels_SD, fp)
    with open(stc_F_file_name, "wb") as fp:   #Pickling
        pickle.dump(con_labels_F, fp)
    with open(stc_O_file_name, "wb") as fp:   #Pickling
        pickle.dump(con_labels_O, fp)
    with open(stc_M_file_name, "wb") as fp:   #Pickling
        pickle.dump(con_labels_M, fp)
    with open(stc_LD_file_name, "wb") as fp:   #Pickling
        pickle.dump(con_labels_LD, fp)
    e=time.time()   
    print(e-s)

if len(sys.argv) == 1:
    # sbj_ids = np.arange(0, len(C.subjects)) 
    sbj_ids = np.array([5,9])
else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]    

for s in sbj_ids:
    SN_functional_connectivity_betweenROIs_runs(s,method)                

