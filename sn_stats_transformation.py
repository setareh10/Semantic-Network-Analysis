#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:27:49 2020

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

from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

# my_stc_coh_SD=[[[[0]*4 for k in range(6)] for w in range(2)] for i in range(18)]
# my_stc_coh_LD=[[[[0]*4 for k in range(6)] for w in range(2)] for i in range(18)]
var_SD=[[]for i in range(18)]
var_LD=[[]for i in range(18)]
ROI_x=1
ROI_y=0
mode='MSE'
for i in np.arange(0,len(C.subjects)):
    
    SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_'+mode+'_SD_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'.json'
    LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/transformation/trans_'+mode+'_LD_x'+str(ROI_x)+'-y'+str(ROI_y)+'_sub_'+str(i)+'.json'
    with open(SD_file_name, "rb") as fp:   # Unpickling
        var_SD[i] = pickle.load(fp)
    
    with open(LD_file_name, "rb") as fp:   # Unpickling
       var_LD[i] = pickle.load(fp)
       
     
##########################################################
for i in np.arange(0,len(C.subjects)-16):
    # if i==0:
    #     var_total_SD= 0.5*(np.log(var_SD[i]+1)-np.log(1-var_SD[i]))
    #     var_total_LD= 0.5*(np.log(var_LD[i]+1)-np.log(1-var_LD[i]))

    # else:
    #     var_total_SD= var_total_SD+ 0.5*(np.log(var_SD[i]+1)-np.log(1-var_SD[i]))
    #     var_total_LD= var_total_LD+ 0.5*(np.log(var_LD[i]+1)-np.log(1-var_LD[i]))
    if i==0:
        var_total_SD= var_SD[i]
        var_total_LD= var_LD[i]

    else:
        var_total_SD= var_total_SD+ var_SD[i]
        var_total_LD= var_total_LD+ var_LD[i]

    
    
plt.plot(np.arange(-0.200,0.5501,0.002),var_SD[i],'b')
plt.plot(np.arange(-0.200,0.5501,0.002),var_LD[i],'g')
plt.plot(np.arange(-0.200,0.5501,0.002),(var_total_SD-var_total_LD),'r')

# ##########################################################
# for i in np.arange(0,len(C.subjects)):
   
#     var_total_SD= 0.5*(np.log(var_SD[i]+1)-np.log(1-var_SD[i]))
#     var_total_LD= 0.5*(np.log(var_LD[i]+1)-np.log(1-var_LD[i]))
#     plt.figure()
#     plt.plot(np.arange(-0.200,0.5501,0.001),(var_total_SD),'b')
#     plt.plot(np.arange(-0.200,0.5501,0.001),var_total_LD,'g')
#     # plt.plot(np.arange(-0.200,0.5501,0.001),(var_total_SD-var_total_LD),'r')
