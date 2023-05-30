#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:43 2020

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



for win in np.arange(0, len(C.con_time_window)):
    tmin = C.con_time_window[win]
    tmax = C.con_time_window[win]+ C.con_time_window_len
    for freq in np.arange(0, len(C.con_freq_band)-1):
        fmin = C.con_freq_band[freq]
        fmax = C.con_freq_band[freq+1]     
        for i in np.arange(0, len(subjects)):
            n_subjects = len(subjects)
            meg = subjects[i]
            sub_to = MRI_sub[i][1:15]

            print('Participant : ' , i, '/ win : ',win, '/ freq : ',freq)
            
            morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                          subject_from='fsaverage',subjects_dir=data_path)
                
        
            # Reading epochs
            epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
            epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
                
            epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
            epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

            
            epochs_SD = epochs_sd['words'].crop(tmin,tmax)
            epochs_LD = epochs_ld['words'].crop(tmin,tmax)
        
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
        
            stc_sd_corrected = [stc_baseline_correction(stc) for stc in stc_sd]
            stc_ld_corrected = [stc_baseline_correction(stc) for stc in stc_ld] 
           
            list_stc_SD = [stc.copy().crop(tmin, tmax) for stc in stc_sd_corrected]
            list_stc_LD = [stc.copy().crop(tmin, tmax) for stc in stc_ld_corrected]
            
            src_SD = inv_op_SD['src']
            src_LD = inv_op_LD['src']

            for j in np.arange(0,len(SN_ROI)):
                morphed_labels[j].subject = sub_to   
            
            label_vertices=[]
            SD=[]
            LD=[]
            for n in np.arange(0,len(list_stc_SD)):
                c=np.zeros([6,251])
                d=np.zeros([6,251])
                for k in np.arange(0,6):
                    a_sd =  list_stc_SD[n].in_label(morphed_labels[k])
                    a_ld =  list_stc_LD[n].in_label(morphed_labels[k])
                    c[k,:]=abs(a_sd.copy().data).mean(0)
                    d[k,:]=abs(a_ld.copy().data).mean(0)
                SD.append(c)
                LD.append(d)

            con_SD,freqs_SD,times_SD,n_epochs_SD,n_tapers_SD=spectral_connectivity(\
                        SD,method='imcoh', mode='fourier',sfreq=C.sfreq,\
                        fmin=fmin,fmax=fmax,n_jobs=3,faverage=True,)   
            
            con_LD,freqs_LD,times_LD,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                        LD,method='imcoh', mode='fourier',sfreq=C.sfreq,\
                        fmin=fmin,fmax=fmax,n_jobs=3,faverage=True,)

            C.im_coh_sd[win, freq, i, :, :] = con_SD.copy().mean(2)
            C.im_coh_ld[win, freq, i, :, :] = con_LD.copy().mean(2)
#         # C.ImCoh_SD_sorted[win,freq,:,:] = sort_matrix(C.ImCoh_SD[win,freq,:,:,:].copy().mean(0),
#         #                              50)
#         # C.ImCoh_LD_sorted[win,freq,:,:] = sort_matrix(C.ImCoh_LD[win,freq,:,:,:].copy().mean(0),
#         #                              50)
        
#         C.ImCoh_SD_LD[win,freq,:,:]= C.ImCoh_SD[win,freq,:,:,:].copy().mean(0)-\
#             C.ImCoh_LD[win,freq,:,:,:].copy().mean(0)
#         C.ImCoh_SD_sorted[win,freq,:,:] = C.ImCoh_SD[win,freq,:,:,:].copy().mean(0)  
#         C.ImCoh_LD_sorted[win,freq,:,:] = C.ImCoh_LD[win,freq,:,:,:].copy().mean(0)  


X_SD = matrix_mirror(np.swapaxes(C.im_coh_sd.copy(), 2, 0))
X_LD = matrix_mirror(np.swapaxes(C.im_coh_ld.copy(), 2, 0))

X = np.zeros([18,8,6,6])
X[:,0,:,:] = X_SD[:,0,0,:,:]
X[:,1,:,:] = X_SD[:,0,1,:,:]
X[:,2,:,:] = X_SD[:,1,0,:,:]
X[:,3,:,:] = X_SD[:,1,1,:,:]
X[:,4,:,:] = X_LD[:,0,0,:,:]
X[:,5,:,:] = X_LD[:,0,1,:,:]
X[:,6,:,:] = X_LD[:,1,0,:,:]
X[:,7,:,:] = X_LD[:,1,1,:,:]


factor_levels = [2,2,2]    
effects = 'all'
Y = X.copy().reshape(18,8,6*6)
fvalss, pvalss = f_mway_rm(Y, factor_levels, effects=effects)
for m in np.arange(0,35,7):
    pvalss[:,m] = 1
    fvalss[:,m] = 0
thresh = mne.stats.f_threshold_mway_rm(18,factor_levels, effects=effects)

effect_labels = ['SD/LD', 'Freq', 'SD/LD by freq','time','SD/LD by time',\
                  'freq by time','all']

# let's visualize our effects by computing f-images
for effect, sig, effect_label in zip(fvalss, pvalss, effect_labels):
    plt.figure()
    # show naive F-values in gray
    plt.imshow(effect.reshape(6, 6), cmap=plt.cm.RdBu_r, extent=[1,
                6, 1, 6], aspect='auto',
                origin='lower')
    # create mask for significant Time-frequency locations
    # effect = np.ma.masked_array(effect, [sig > .05])
    # plt.imshow(effect.reshape(6, 6), cmap='RdBu_r', extent=[1,
    #            6, 1, 6], aspect='auto',
    #            origin='lower')
    plt.colorbar()
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title(r"Time-locked response for '%s' "% (effect_label) )
    plt.show()
# # con_SD_LD = dict()
# # Con_SD = dict()
# # Con_LD = dict()


con_SD_LD['imcoh'] = C.im_coh_sd_ld
Con_SD['imcoh'] = C.im_coh_sd_sorted
Con_LD['imcoh'] = C.im_coh_ld_sorted
vmax_SD_LD = max(abs(C.im_coh_sd_ld.max()), abs(C.im_coh_sd_ld.min()))
vmax_SD = max(abs(C.im_coh_sd.max()), abs(C.im_coh_sd.min()))
vmax_LD = max(abs(C.im_coh_ld.max()), abs(C.im_coh_ld.min()))
vmax = max(vmax_SD_LD,vmax_SD,vmax_LD)
vmin = -vmax

for win in np.arange(0, len(C.con_time_window)):
    tmin = C.con_time_window[win]
    tmax = C.con_time_window[win]+ C.con_time_window_len
    for freq in np.arange(0, len(C.con_freq_band)-1):
  
        fmin = C.con_freq_band[freq]
        fmax = C.con_freq_band[freq+1]             
        con_SD_LD['imcoh'] = C.im_coh_sd_ld[win, freq, :, :]
        Con_SD['imcoh'] = C.im_coh_sd_sorted[win, freq, :, :]
        Con_LD['imcoh'] = C.im_coh_ld_sorted[win, freq, :, :]
        
        # labels = mne.read_labels_from_annot( 'fsaverage', parc='aparc',
        #                                     subjects_dir=C.data_path)
        # unknwon = labels.pop(68)
        
        # label_colors = [label.color for label in SN_ROI] 
        label_colors = [(0.06,0.53,0.69,1),(0.02,0.83,0.62,1),\
                        (0.02,0.23,0.29,1),(0.93,0.27,0.43,1),\
                            (1,0.81,0.4,1),(0.06,0.53,0.69,1)]   

        # label_names = [label.name for label in SN_ROI]
        label_names = ['lATL','MTG_ITG','IFG','AG','PVA','rATL']
    
        lh_labels = ['lATL','MTG_ITG','IFG','AG','PVA']
        rh_labels = ['rATL']


        node_order = ['lATL','MTG_ITG','IFG','AG','PVA','rATL']
        node_angles = circular_layout(label_names, node_order, start_pos=90,
                                      group_boundaries=[0, len(label_names) / 2])
        
      
        fig_con, axes_con = plot_connectivity_circle(Con_SD['imcoh'], label_names, n_lines=None,\
                          node_angles=node_angles, node_colors=label_colors,\
                          title='Connectivity (ImCoh) SD \n'+\
                          f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'ms, '+f'{fmin}' +'-'+\
                          f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
                          vmin=-vmax_SD, vmax=vmax_SD,colormap='RdBu')  
        fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD_'+\
                          f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
                          f'{fmax}_rdbu.jpg')
            
        fig_con, axes_con = plot_connectivity_circle(Con_LD['imcoh'], label_names, n_lines=None,\
                          node_angles=node_angles, node_colors=label_colors,\
                          title='Connectivity (ImCoh) LD \n'+\
                          f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'ms, '+f'{fmin}' +'-'+\
                          f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
                          vmin=-vmax_LD, vmax=vmax_LD,colormap='RdBu')  
        fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)LD_'+\
                          f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
                          f'{fmax}_rdbu.jpg')                         
                                
        fig_con, axes_con = plot_connectivity_circle(con_SD_LD['imcoh'], label_names, n_lines=None,\
                          node_angles=node_angles, node_colors=label_colors,\
                          title='Connectivity (ImCoh) SD-LD \n'+\
                          f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'ms, '+f'{fmin}' +'-'+\
                          f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
                          vmin=-vmax_SD_LD, vmax=vmax_SD_LD,colormap='RdBu')  
        fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD-LD_'+\
                          f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
                          f'{fmax}_rdbu.jpg')    
        
# # plt.close('all')

#RdBu

        